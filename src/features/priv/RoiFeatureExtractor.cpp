#include "RoiFeatureExtractor.hpp"
#include "LegacyRoiFeatureExtractor.hpp"
#include "ImageFeatureExtractor.hpp"
#include "RoiEmbeddingCore.hpp"
#include <inferrt/core/Exception.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <utility>
#include <iostream>

namespace irt::features::priv {
namespace {
embedding::Shape shapeOf(const RoiFeatureItem& item) {
    embedding::Shape s{{item.roi.x1,item.roi.y1,item.roi.x2,item.roi.y2},{}};
    s.polygon.reserve(item.polygon.size());
    for (auto p:item.polygon) s.polygon.push_back({p.x,p.y});
    return s;
}
cv::Mat decodeImage(const std::filesystem::path& path) {
    // filesystem::path preserves wide Windows paths; no temporary crop files.
    std::ifstream in(path, std::ios::binary);
    if (!in) throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,"Cannot read ROI image");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),{});
    cv::Mat image=cv::imdecode(bytes,cv::IMREAD_COLOR);
    if(image.empty()) throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,"Cannot decode ROI image");
    return image;
}
int interpolation(irt::Interpolation value) {
    switch(value) {
        case irt::Interpolation::Nearest:return cv::INTER_NEAREST;
        case irt::Interpolation::Cubic:return cv::INTER_CUBIC;
        case irt::Interpolation::Area:return cv::INTER_AREA;
        default:return cv::INTER_LINEAR;
    }
}
void appendRaster(const cv::Mat& image,const embedding::View& v,const std::vector<float>& mask,
                  const irt::PreprocessSpec& pre,float background_keep,std::vector<float>& input) {
    cv::Mat resized;
    cv::resize(image(cv::Rect(v.x,v.y,v.width,v.height)),resized,
               cv::Size(v.resized_width,v.resized_height),0,0,interpolation(pre.interpolation));
    size_t plane=size_t(v.output_width)*v.output_height, begin=input.size();
    input.resize(begin+3*plane,0.f); // zero after normalization = channel mean before normalization
    for(int y=0;y<v.resized_height;++y) {
        const auto* row=resized.ptr<cv::Vec3b>(y);
        for(int x=0;x<v.resized_width;++x) {
            size_t dst=size_t(y+v.top)*v.output_width+x+v.left;
            float alpha=background_keep+(1-background_keep)*mask[dst];
            for(int c=0;c<3;++c) {
                int src=pre.dst_color==irt::ColorFormat::RGB?2-c:c;
                // Blending with the model mean in raw space is alpha * normalized value.
                input[begin+size_t(c)*plane+dst]=alpha*(float(row[x][src])*pre.scale-pre.mean[c])/pre.stddev[c];
            }
        }
    }
}
}

class RoiFeatureExtractor::Impl {
public:
    Impl(const RoiFeatureConfig& config,const std::filesystem::path& weights):config_(config) {
        if(config_.mode==RoiFeatureMode::LegacyRoiAlign) {
            legacy_=std::make_unique<LegacyRoiFeatureExtractor>(config_,weights);return;
        }
        embedding::require(!config_.use_pca,"Per-image PCA is not a shared semantic space; disable use_pca");
        embedding::require(config_.norm==ImageSearchFeatureNorm::L2,"Semantic ROI features require L2 norm");
        embedding::require(std::isfinite(config_.background_keep)&&config_.background_keep>=0&&config_.background_keep<=1,
                           "background_keep must be in [0,1]");
        embedding::require(std::isfinite(config_.spatial_weight)&&config_.spatial_weight>=0&&config_.spatial_weight<1,
                           "spatial_weight must be in [0,1)");
        embedding::require(std::isfinite(config_.detail_weight)&&config_.detail_weight>=0&&config_.detail_weight<1,
                           "detail_weight must be in [0,1)");
        embedding::require(config_.max_detail_views==0||config_.max_detail_views==2||config_.max_detail_views==3,
                           "max_detail_views must be 0, 2 or 3");
        embedding::require(config_.feature_name=="x_norm_patchtokens","Semantic mode requires DINO patch tokens only");
        stats_.available=true;
        ImageSearchConfig base=config_;
        const int default_edge=config_.patch_size==14?518:512;
        if(base.preprocess.input_width==0&&base.preprocess.input_height==0) {
            base.preprocess.input_width=default_edge;base.preprocess.input_height=default_edge;
        }
        image_=std::make_unique<ImageFeatureExtractor>(config_.model_name,config_.feature_name,weights,base,true);
        w_=image_->inputWidth();h_=image_->inputHeight();
        auto dims=image_->featureTensorShape();
        embedding::require(config_.patch_size>0&&w_%config_.patch_size==0&&h_%config_.patch_size==0,
                           "Model input must be divisible by DINO patch_size");
        gh_=h_/config_.patch_size;gw_=w_/config_.patch_size;
        embedding::require(dims.nbDims==3&&dims.d[1]==gh_*gw_&&dims.d[2]>0,
                           "Expected B x (H/patch * W/patch) x D; exclude CLS/register tokens");
        d_=int(dims.d[2]);
        const auto& pre=image_->preprocessSpec();
        embedding::require(pre.input_channels==3&&(pre.dst_color==irt::ColorFormat::RGB||pre.dst_color==irt::ColorFormat::BGR),
                           "Semantic crop path requires three-channel RGB or BGR model input");
    }
    int featureDim() const {return legacy_?legacy_->featureDim():d_*(config_.spatial_weight>0?5:1);}
    RoiFeatureWorkStats workStats() const noexcept {return stats_;}
    size_t maxBatchSize() const noexcept{return legacy_?legacy_->maxBatchSize():image_->maxBatchSize();}
    std::vector<float> extract(const RoiFeatureItem& item){return extractItems({item});}
    std::vector<float> extractBatch(const std::vector<RoiFeatureItem>& items,size_t begin,size_t count) {
        embedding::require(begin<=items.size()&&count<=items.size()-begin,"Invalid ROI batch range");
        return extractItems(std::vector<RoiFeatureItem>(items.begin()+begin,items.begin()+begin+count));
    }
    std::vector<float> extractItems(const std::vector<RoiFeatureItem>& items) {
        if(legacy_)return legacy_->extractItems(items);
        return extractAll(items,{});
    }
    std::vector<float> extractAll(const std::vector<RoiFeatureItem>& items,
                                 const std::function<void(size_t,size_t,size_t,size_t)>& progress) {
        const size_t dim=size_t(featureDim());
        std::vector<float> result(irt::checkedSizeMul(items.size(),dim,"ROI output elements"));
        extractTo(items,[&](const std::vector<size_t>& rows,const std::vector<float>& values) {
            embedding::scatterRows(rows,values,dim,result.data(),items.size());
        },progress);
        return result;
    }
    void extractTo(const std::vector<RoiFeatureItem>& items,const RoiFeatureExtractor::BatchConsumer& consume,
                   const std::function<void(size_t,size_t,size_t,size_t)>& progress) {
        if(legacy_) {
            auto features=legacy_->extractAll(items,progress);
            std::vector<size_t> rows(items.size());std::iota(rows.begin(),rows.end(),0);
            if(!items.empty())consume(rows,features);
            return;
        }
        const size_t dim=size_t(featureDim());
        std::map<std::filesystem::path,std::vector<size_t>> groups;
        for(size_t i=0;i<items.size();++i)groups[items[i].image_path].push_back(i);
        struct Pending {size_t roi;bool last;double area;std::vector<float> weights;};
        struct Accum {std::vector<std::vector<float>> descriptors;std::vector<double> areas;};
        std::vector<Pending> pending;
        std::vector<float> input;
        input.reserve(maxBatchSize()*size_t(3)*w_*h_);
        std::map<size_t,Accum> accum;
        size_t processed=0,batch_index=0;
        auto flush=[&] {
            if(pending.empty())return;
            ++stats_.forward_batches;stats_.encoded_views+=pending.size();
            auto tensor=image_->extractPreparedTensorBatch(input,pending.size());
            input.clear();input.reserve(maxBatchSize()*size_t(3)*w_*h_);
            embedding::require(tensor.dims.nbDims==3&&tensor.dims.d[0]==int64_t(pending.size())
                               &&tensor.dims.d[1]==gh_*gw_&&tensor.dims.d[2]==d_,"Runtime patch tensor shape changed");
            size_t completed=0,first=0;
            std::vector<size_t> ready_rows;ready_rows.reserve(pending.size());
            std::vector<float> ready_values;ready_values.reserve(pending.size()*dim);
            for(size_t i=0;i<pending.size();++i) {
                auto& job=pending[i];auto& a=accum[job.roi];
                a.descriptors.push_back(embedding::pool(tensor.data.data()+i*size_t(gh_)*gw_*d_,gh_,gw_,d_,job.weights,
                                                        config_.spatial_weight));
                a.areas.push_back(job.area);
                if(job.last) {
                    auto descriptor=embedding::fuse(a.descriptors,a.areas,config_.detail_weight);
                    ready_rows.push_back(job.roi);
                    ready_values.insert(ready_values.end(),descriptor.begin(),descriptor.end());
                    accum.erase(job.roi);if(completed==0)first=job.roi;++completed;
                }
            }
            if(!ready_rows.empty())consume(ready_rows,ready_values);
            processed+=completed;
            if(progress&&completed)progress(batch_index++,first,completed,processed);
            pending.clear();
        };
        for(const auto& group:groups) {
            const cv::Mat image=decodeImage(group.first); // one decoded source at a time
            ++stats_.decoded_images;
            for(auto index:group.second) {
                auto shape=shapeOf(items[index]);
                auto views=embedding::planViews(shape,image.cols,image.rows,w_,h_,config_.patch_size,
                                               config_.crop_margin,config_.max_detail_views);
                struct Job {embedding::View view;std::vector<float> mask,weights;double area;};
                std::vector<Job> jobs;
                for(size_t vi=0;vi<views.size();++vi) {
                    auto mask=embedding::rasterMask(shape,views[vi]);
                    auto weights=embedding::patchWeights(mask,w_,h_,config_.patch_size);
                    double mass=std::accumulate(weights.begin(),weights.end(),0.);
                    if(mass<=1e-8) {
                        embedding::require(vi!=0,"ROI has no rasterized support inside the image");continue;
                    }
                    double area=mass*config_.patch_size*config_.patch_size/(views[vi].sx*views[vi].sy);
                    jobs.push_back({views[vi],std::move(mask),std::move(weights),area});
                }
                for(size_t j=0;j<jobs.size();++j) {
                    auto& job=jobs[j];
                    appendRaster(image,job.view,job.mask,image_->preprocessSpec(),config_.background_keep,input);
                    pending.push_back({index,j+1==jobs.size(),job.area,std::move(job.weights)});
                    if(pending.size()==maxBatchSize())flush();
                }
            }
            // Pending inputs own their pixels, so the source image can be released here.
            // Keep filling the batch from the next image instead of forcing batch=1.
        }
        flush();
    }
private:
    RoiFeatureConfig config_;
    RoiFeatureWorkStats stats_;
    std::unique_ptr<LegacyRoiFeatureExtractor> legacy_;
    std::unique_ptr<ImageFeatureExtractor> image_;
    int w_{},h_{},gh_{},gw_{},d_{};
};
RoiFeatureExtractor::RoiFeatureExtractor(const RoiFeatureConfig& config,const std::filesystem::path& weights)
    :impl_(std::make_unique<Impl>(config,weights)){}
RoiFeatureExtractor::~RoiFeatureExtractor()=default;
RoiFeatureExtractor::RoiFeatureExtractor(RoiFeatureExtractor&&) noexcept=default;
RoiFeatureExtractor& RoiFeatureExtractor::operator=(RoiFeatureExtractor&&) noexcept=default;
RoiFeatureWorkStats RoiFeatureExtractor::workStats()const noexcept{return impl_->workStats();}
int RoiFeatureExtractor::featureDim()const{return impl_->featureDim();}
size_t RoiFeatureExtractor::maxBatchSize()const noexcept{return impl_->maxBatchSize();}
std::vector<float> RoiFeatureExtractor::extract(const RoiFeatureItem& item){return impl_->extract(item);}
std::vector<float> RoiFeatureExtractor::extractItems(const std::vector<RoiFeatureItem>& items){return impl_->extractItems(items);}
std::vector<float> RoiFeatureExtractor::extractBatch(const std::vector<RoiFeatureItem>& items,size_t begin,size_t count){return impl_->extractBatch(items,begin,count);}
std::vector<float> RoiFeatureExtractor::extractAll(const std::vector<RoiFeatureItem>& items,
    const std::function<void(size_t,size_t,size_t,size_t)>& progress){return impl_->extractAll(items,progress);}
void RoiFeatureExtractor::extractTo(const std::vector<RoiFeatureItem>& items,const BatchConsumer& consume,
    const std::function<void(size_t,size_t,size_t,size_t)>& progress){impl_->extractTo(items,consume,progress);}
} // namespace irt::features::priv
