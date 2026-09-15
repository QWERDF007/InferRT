#pragma once

// Frozen-feature retrieval algorithms. No inference/runtime dependencies.
// Coordinates always describe the continuous ROI, never its enclosing patch block.
#include "DinoTypes.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace irt::features::priv::retrieval {

inline float dot(const float *a, const float *b, int d) {
    float s = 0; for (int i=0; i<d; ++i) s += a[i]*b[i]; return s;
}
inline bool normalize(std::vector<float>& v) {
    double s=0; for(float x:v) s+=double(x)*x;
    if (!(s>1e-20) || !std::isfinite(s)) return false;
    float f=float(1/std::sqrt(s)); for(float& x:v) x*=f; return true;
}
inline double iou(const DinoRect& a,const DinoRect& b) {
    double z=DinoRect::intersect(a,b).area(); double u=a.area()+b.area()-z;
    return u>0 ? z/u : 0;
}
inline DinoRect centered(double x,double y,double w,double h) {
    return {x-w*.5,y-h*.5,x+w*.5,y+h*.5};
}

// A deterministic subsampled randomized Hadamard transform. No fitting/codebook.
// D==d is the lossless ablation. Padded transform rows are permuted, not just truncated.
class Projection {
    int input_, output_, padded_{1};
    std::vector<float> signs_;
    std::vector<int> rows_;
public:
    Projection(int input,int output):input_(input),output_(output) {
        if(input<=0 || output<=0 || output>input) throw std::invalid_argument("projection dimensions");
        while(padded_<input) padded_*=2;
        uint32_t state=20260915U;
        auto next=[&](){state^=state<<13;state^=state>>17;state^=state<<5;return state;};
        signs_.resize(input); for(auto& x:signs_) x=(next()&1U)?1.f:-1.f;
        rows_.resize(padded_); std::iota(rows_.begin(),rows_.end(),0);
        for(int i=padded_-1;i>0;--i) std::swap(rows_[i],rows_[next()%uint32_t(i+1)]);
    }
    std::vector<float> apply(const float *v) const {
        if(input_==output_) return std::vector<float>(v,v+input_);
        std::vector<float> work(padded_,0.f);
        for(int i=0;i<input_;++i) work[i]=v[i]*signs_[i];
        for(int step=1;step<padded_;step*=2)
            for(int base=0;base<padded_;base+=2*step)
                for(int j=0;j<step;++j) {
                    float a=work[base+j],b=work[base+j+step];
                    work[base+j]=a+b;work[base+j+step]=a-b;
                }
        std::vector<float> result(output_);
        for(int i=0;i<output_;++i) result[i]=work[rows_[i]];
        if(!normalize(result)) throw std::runtime_error("zero projected descriptor; use original dimension");
        return result;
    }
};

struct Sample { int patch{-1},cell{0}; float weight{0}; DinoPoint point{}; };

// A whole valid grid and 3x3 overlapping half-size windows. Unlike windows based
// on the short side, this also covers wide/tall whole-image views without excess windows.
inline std::vector<DinoRegionDescriptor> regionWindows(const DinoFeatureGrid& grid,int view_id,float min_valid) {
    int y0=grid.plan.grid_height,x0=grid.plan.grid_width,y1=0,x1=0;
    for(int y=0;y<grid.plan.grid_height;++y)for(int x=0;x<grid.plan.grid_width;++x)
        if(grid.patchValid(y,x)){y0=std::min(y0,y);x0=std::min(x0,x);y1=std::max(y1,y+1);x1=std::max(x1,x+1);}
    std::vector<DinoRegionDescriptor> result;if(y1<=y0||x1<=x0)return result;
    auto append=[&](int y,int x,int h,int w){
        for(const auto& r:result)if(r.grid_row==y&&r.grid_col==x&&r.grid_height==h&&r.grid_width==w)return;
        float valid=0;for(int r=y;r<y+h;++r)for(int c=x;c<x+w;++c)valid+=grid.valid_area[size_t(r*grid.plan.grid_width+c)];
        valid/=float(h*w);if(valid<min_valid)return;
        DinoRegionDescriptor d;d.view_id=view_id;d.grid_row=y;d.grid_col=x;d.grid_height=h;d.grid_width=w;d.valid_fraction=valid;
        d.source_rect=DinoRect::unite(grid.plan.patchRect(y,x),grid.plan.patchRect(y+h-1,x+w-1));result.push_back(d);
    };
    append(y0,x0,y1-y0,x1-x0);
    int h=std::max(1,(y1-y0+1)/2),w=std::max(1,(x1-x0+1)/2);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)append(y0+(y1-y0-h)*i/2,x0+(x1-x0-w)*j/2,h,w);
    return result;
}

// Each patch is owned by one cell. Fractional ROI overlap survives even when no
// patch centre lies inside the ROI. Multiple samples share, never duplicate, cell weight.
inline std::vector<Sample> select(const DinoFeatureGrid& grid,const DinoRect& roi,
                                  const std::vector<float>& weights,int cells,int per_cell) {
    if(roi.empty() || cells<1 || per_cell<1) return {};
    const double px=grid.plan.sourcePxPerPatchX(), py=grid.plan.sourcePxPerPatchY();
    int cols=std::min(cells,std::max(1,int(std::ceil(roi.width()/std::max(px,1e-12)))));
    int rows=std::min(cells,std::max(1,int(std::ceil(roi.height()/std::max(py,1e-12)))));
    std::vector<std::vector<Sample>> bins(size_t(rows*cols));
    for(int r=0;r<grid.plan.grid_height;++r) for(int c=0;c<grid.plan.grid_width;++c) {
        int p=r*grid.plan.grid_width+c;
        if(!(weights[p]>0) || !grid.patchValid(r,c)) continue;
        auto overlap=DinoRect::intersect(grid.plan.patchRect(r,c),roi);
        if(overlap.empty()) continue;
        DinoPoint point{(overlap.x0+overlap.x1)*.5,(overlap.y0+overlap.y1)*.5};
        int bc=std::clamp(int((point.x-roi.x0)/roi.width()*cols),0,cols-1);
        int br=std::clamp(int((point.y-roi.y0)/roi.height()*rows),0,rows-1);
        bins[size_t(br*cols+bc)].push_back({p,br*cols+bc,weights[p],point});
    }
    std::vector<Sample> result;
    for(size_t cell=0;cell<bins.size();++cell) {
        auto& bin=bins[cell]; if(bin.empty()) continue;
        double cx=roi.x0+(double(cell%cols)+.5)*roi.width()/cols;
        double cy=roi.y0+(double(cell/cols)+.5)*roi.height()/rows;
        auto distance=[&](const Sample& s){double x=(s.point.x-cx)/px,y=(s.point.y-cy)/py;return x*x+y*y;};
        size_t first=0; float weight=0;
        for(size_t i=0;i<bin.size();++i){weight+=bin[i].weight;if(distance(bin[i])<distance(bin[first]))first=i;}
        std::vector<size_t> chosen{first};
        std::vector<float> min_distance(bin.size(),4.f);
        while(chosen.size()<size_t(per_cell) && chosen.size()<bin.size()) {
            auto last=chosen.back(); size_t best=bin.size(); float far=-1;
            const float* a=grid.tokens.data()+size_t(bin[last].patch)*grid.channels;
            for(size_t j=0;j<bin.size();++j) {
                const float* b=grid.tokens.data()+size_t(bin[j].patch)*grid.channels;
                min_distance[j]=std::min(min_distance[j],std::max(0.f,2.f-2.f*dot(a,b,grid.channels)));
                if(std::find(chosen.begin(),chosen.end(),j)!=chosen.end()) continue;
                if(min_distance[j]>far){far=min_distance[j];best=j;}
            }
            if(best==bin.size()) break;
            chosen.push_back(best);
        }
        for(size_t j:chosen){auto s=bin[j];s.weight=weight/float(chosen.size());result.push_back(s);}
    }
    float total=0;for(const auto& s:result)total+=s.weight;
    if(total>0)for(auto& s:result)s.weight/=total;
    return result;
}

struct Evidence { DinoPoint point{}; int cell{0}; float weight{0}; };
struct Match { int index{-1}; float score{-2.f}; };
struct Pair { Match first{},second{}; };
inline void insert(Pair& pair, Match m) {
    if(m.index<0 || !std::isfinite(m.score)) return;
    auto better=[](Match a,Match b){return a.score>b.score || (a.score==b.score && (b.index<0 || a.index<b.index));};
    if(pair.first.index==m.index){if(better(m,pair.first))pair.first=m;return;}
    if(better(m,pair.first)){pair.second=pair.first;pair.first=m;}
    else if(pair.second.index!=m.index && better(m,pair.second)) pair.second=m;
}

inline std::vector<Pair> nearestTwo(const float* gallery,size_t count,const float* query,size_t q,int dim) {
    std::vector<Pair> pairs(q);
    for(size_t t=0;t<q;++t)for(size_t g=0;g<count;++g)
        insert(pairs[t],{int(g),dot(query+t*dim,gallery+g*dim,dim)});
    return pairs;
}

struct Pose { DinoRect box{}; float score{0},appearance{0},support{0}; };

// Bounded translation/scale voting for ALL gallery views, before global Top-K.
// A cell contributes its weight at most once per bin; no ratio-test hard rejection.
inline std::vector<Pose> vote(const std::vector<Evidence>& evidence,const std::vector<Pair>& matches,
                              const std::vector<DinoPoint>& points,const DinoRect& roi,
                              double base_scale,double gallery_patch,int keep=2) {
    if(evidence.empty() || evidence.size()!=matches.size() || roi.empty()) return {};
    struct Vote {int level;int64_t x,y;int cell;float similarity,weight;double cx,cy,scale;};
    std::vector<Vote> votes; votes.reserve(evidence.size()*6);
    double rx=(roi.x0+roi.x1)*.5,ry=(roi.y0+roi.y1)*.5;
    float total_weight=0; std::vector<float> cell_weights;
    for(const auto& e:evidence){if(size_t(e.cell)>=cell_weights.size())cell_weights.resize(size_t(e.cell)+1,0);cell_weights[e.cell]+=e.weight;total_weight+=e.weight;}
    if(!(total_weight>0))return {};
    for(int level=0;level<3;++level){
        double scale=base_scale*std::pow(2.,.5*(level-1));
        double bin=std::max(1.,std::min(gallery_patch*2.,std::min(roi.width(),roi.height())*scale*.5));
        for(size_t t=0;t<evidence.size();++t){
            const auto& e=evidence[t];
            for(const auto& m:{matches[t].first,matches[t].second}){
                if(m.index<0 || size_t(m.index)>=points.size())continue;
                const auto& p=points[size_t(m.index)];
                double cx=p.x+scale*(rx-e.point.x),cy=p.y+scale*(ry-e.point.y);
                votes.push_back({level,int64_t(std::llround(cx/bin)),int64_t(std::llround(cy/bin)),e.cell,
                                 std::clamp((m.score+1)*.5f,0.f,1.f),cell_weights[e.cell],cx,cy,scale});
            }
        }
    }
    std::sort(votes.begin(),votes.end(),[](const Vote& a,const Vote& b){
        if(std::tie(a.level,a.x,a.y,a.cell)!=std::tie(b.level,b.x,b.y,b.cell))
            return std::tie(a.level,a.x,a.y,a.cell)<std::tie(b.level,b.x,b.y,b.cell);
        return a.similarity>b.similarity;
    });
    std::vector<Pose> poses;
    for(size_t begin=0;begin<votes.size();){
        size_t end=begin;double sx=0,sy=0,squared=0;float w=0,a=0;int last_cell=-1;
        const double origin_x=votes[begin].cx,origin_y=votes[begin].cy;
        while(end<votes.size() && std::tie(votes[end].level,votes[end].x,votes[end].y)==std::tie(votes[begin].level,votes[begin].x,votes[begin].y)){
            const auto& v=votes[end++];if(v.cell==last_cell)continue;last_cell=v.cell;
            const double dx=v.cx-origin_x,dy=v.cy-origin_y;
            w+=v.weight;a+=v.weight*v.similarity;sx+=dx*v.weight;sy+=dy*v.weight;
            squared+=(dx*dx+dy*dy)*v.weight;
        }
        if(w>0){float app=a/total_weight,support=w/total_weight;
            const double variance=std::max(0.,squared/w-(sx*sx+sy*sy)/(w*w));
            const double bin=std::max(1.,std::min(gallery_patch*2.,std::min(roi.width(),roi.height())*votes[begin].scale*.5));
            const float agreement=float(std::max(0.,1.-std::sqrt(variance)/bin));
            poses.push_back({centered(origin_x+sx/w,origin_y+sy/w,roi.width()*votes[begin].scale,roi.height()*votes[begin].scale),
                             .6f*app+.25f*support+.15f*support*agreement,app,support});}
        begin=end;
    }
    std::stable_sort(poses.begin(),poses.end(),[](const Pose&a,const Pose&b){return a.score>b.score;});
    std::vector<Pose> kept;
    for(const auto& p:poses){bool duplicate=false;for(const auto& k:kept)if(iou(k.box,p.box)>.5)duplicate=true;
        if(!duplicate)kept.push_back(p);
        if(int(kept.size())>=keep)break;
    }
    return kept;
}

struct Located {DinoRect box{};float score{-1},appearance{0},coverage{0},consistency{0};};
struct FineOptions {
    int max_sizes{12},peaks{3},refinement_rounds{1};
    double scale_step{1.4142135623730951},min_short{.75},nms{.5};
    float match_cosine{.55f},appearance_weight{.6f},coverage_weight{.25f},consistency_weight{.15f};
};

// S[q, gallery patch] is evaluated once in the original feature dimension.
// The geometric sweep only reads scalar maps. The output box is the continuous ROI.
class SparseMatcher {
    int h_,w_; std::vector<Evidence> evidence_;DinoRect roi_;
    std::vector<float> sim_,near_;std::vector<int> near_index_;std::vector<float> valid_;
    static float interpolate(const float* a,int h,int w,double x,double y) {
        x=std::clamp(x,0.,double(w-1));y=std::clamp(y,0.,double(h-1));
        int x0=int(x),y0=int(y),x1=std::min(w-1,x0+1),y1=std::min(h-1,y0+1);
        float dx=float(x-x0),dy=float(y-y0);
        return (1-dy)*((1-dx)*a[y0*w+x0]+dx*a[y0*w+x1])+dy*((1-dx)*a[y1*w+x0]+dx*a[y1*w+x1]);
    }
public:
    SparseMatcher(int h,int w,std::vector<Evidence> evidence,DinoRect roi,std::vector<float> similarities,std::vector<float> valid)
        :h_(h),w_(w),evidence_(std::move(evidence)),roi_(roi),sim_(std::move(similarities)),valid_(std::move(valid)) {
        if(h<=0||w<=0||roi.empty()||sim_.size()!=evidence_.size()*size_t(h*w)||valid_.size()!=size_t(h*w))
            throw std::invalid_argument("sparse matcher dimensions");
        near_.assign(sim_.size(),-2.f);near_index_.assign(sim_.size(),-1);
        for(size_t t=0;t<evidence_.size();++t)for(int y=0;y<h;++y)for(int x=0;x<w;++x){
            size_t out=t*size_t(h*w)+size_t(y*w+x);
            for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                int xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=w||yy>=h||!(valid_[yy*w+xx]>0))continue;
                float s=sim_[t*size_t(h*w)+size_t(yy*w+xx)];
                if(s>near_[out]){near_[out]=s;near_index_[out]=yy*w+xx;}
            }
        }
    }
    Located evaluate(const DinoRect& b,const FineOptions& o) const {
        Located result;result.box=b;
        if(b.empty()||b.x0<0||b.y0<0||b.x1>w_||b.y1>h_)return result;
        double total=0,a=0,c=0,g=0;
        for(size_t t=0;t<evidence_.size();++t){
            const auto& e=evidence_[t]; total+=e.weight;
            double x=b.x0+(e.point.x-roi_.x0)/roi_.width()*b.width()-.5;
            double y=b.y0+(e.point.y-roi_.y0)/roi_.height()*b.height()-.5;
            int xx=std::clamp(int(std::lround(x)),0,w_-1),yy=std::clamp(int(std::lround(y)),0,h_-1);
            size_t p=size_t(yy*w_+xx), offset=t*size_t(h_*w_);
            float validity=interpolate(valid_.data(),h_,w_,x,y);
            if(!(validity>0))continue;
            // Invalid padding contributes zero evidence, never renormalizes the query away.
            float s=interpolate(sim_.data()+offset,h_,w_,x,y);
            a+=e.weight*validity*std::clamp((s+1)*.5f,0.f,1.f);
            if(near_[offset+p]>=o.match_cosine){
                int n=near_index_[offset+p];double distance=std::hypot(double(n%w_)-x,double(n/w_)-y);
                c+=e.weight*validity;g+=e.weight*validity*std::max(0.,1.-distance/1.5);
            }
        }
        if(total>0){result.appearance=float(a/total);result.coverage=float(c/total);result.consistency=float(g/total);
            result.score=o.appearance_weight*result.appearance+o.coverage_weight*result.coverage+o.consistency_weight*result.consistency;}
        return result;
    }
    std::vector<Located> locate(const FineOptions& o,const DinoRect& hint={},const std::function<bool()>& expired={}) const {
        std::vector<Located> beam;
        auto retain=[&](Located p){
            if(p.score<0)return;
            for(const auto& kept:beam) if(iou(kept.box,p.box)>o.nms && kept.score>=p.score)return;
            beam.erase(std::remove_if(beam.begin(),beam.end(),[&](const Located& kept){return iou(kept.box,p.box)>o.nms;}),beam.end());
            beam.push_back(p);std::stable_sort(beam.begin(),beam.end(),[](auto&a,auto&b){return a.score>b.score;});
            if(int(beam.size())>o.peaks)beam.resize(size_t(o.peaks));
        };
        if(!hint.empty())retain(evaluate(hint,o));
        double ratio=roi_.width()/roi_.height();
        for(int level=0;level<o.max_sizes;++level){
            if(expired && expired())break;
            double short_edge=o.min_short*std::pow(o.scale_step,level);
            for(double aspect:{.7071067811865475,1.,1.4142135623730951}){
                double r=ratio*aspect,bw=short_edge*std::max(1.,r),bh=short_edge*std::max(1.,1./r);
                if(bw>w_||bh>h_)continue;
                // Include both valid boundary positions; no systematic last-row/column miss.
                int nx=int(std::ceil(w_-bw)),ny=int(std::ceil(h_-bh));
                for(int iy=0;iy<=ny;++iy)for(int ix=0;ix<=nx;++ix){
                    double x=std::min(double(ix),w_-bw),y=std::min(double(iy),h_-bh);
                    retain(evaluate({x,y,x+bw,y+bh},o));
                }
            }
        }
        auto seeds=beam;
        for(auto p:seeds)for(int round=0;round<o.refinement_rounds;++round){
            if(expired && expired())break;
            auto base=p;
            for(double sy:{.8408964152537145,1.,1.189207115002721})for(double sx:{.8408964152537145,1.,1.189207115002721})
                for(double dy:{-.5,0.,.5})for(double dx:{-.5,0.,.5}){
                    auto b=centered((base.box.x0+base.box.x1)*.5+dx,(base.box.y0+base.box.y1)*.5+dy,base.box.width()*sx,base.box.height()*sy);
                    auto candidate=evaluate(b,o);if(candidate.score>p.score)p=candidate;
                }
            retain(p);
        }
        return beam;
    }
};

inline std::vector<float> similarityMatrix(const float* q,size_t query_count,const DinoFeatureGrid& grid) {
    size_t patches=size_t(grid.plan.patchCount());std::vector<float> s(query_count*patches);
    for(size_t t=0;t<query_count;++t)for(size_t p=0;p<patches;++p)
        s[t*patches+p]=grid.valid_area[p]>0?dot(q+t*grid.channels,grid.tokens.data()+p*grid.channels,grid.channels):-1.f;
    return s;
}

} // namespace irt::features::priv::retrieval
