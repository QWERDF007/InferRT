#pragma once
// Geometry and pooling shared by the production extractor and dependency-free tests.
// No model, image decoder, fitting, or persistent patch feature storage here.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace irt::features::embedding {
struct Point { double x{}, y{}; };
struct Box { double x1{}, y1{}, x2{}, y2{}; };
struct Shape { Box box; std::vector<Point> polygon; };
struct View {
    int x{}, y{}, width{}, height{}; // integer original-image crop
    int output_width{}, output_height{}, resized_width{}, resized_height{}, left{}, top{};
    double sx{}, sy{};
    Point map(Point p) const { return {(p.x-x)*sx+left, (p.y-y)*sy+top}; }
};
inline void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
// Scatter a completed crop batch into its final vector storage. Shared by the
// matrix-returning API and the streaming Faiss builder; no full-matrix staging.
inline void scatterRows(const std::vector<size_t>& rows,const std::vector<float>& values,
                        size_t dim,float* destination,size_t total_rows) {
    require(dim>0&&values.size()/dim==rows.size()&&values.size()%dim==0,"Feature batch shape mismatch");
    require(rows.empty()||destination,"Missing feature destination");
    for(size_t i=0;i<rows.size();++i) {
        require(rows[i]<total_rows,"Feature row index is out of range");
        std::copy_n(values.data()+i*dim,dim,destination+rows[i]*dim);
    }
}
inline Box bounds(const Shape& s) {
    if (s.polygon.empty()) {
        const auto b=s.box;
        require(std::isfinite(b.x1)&&std::isfinite(b.y1)&&std::isfinite(b.x2)&&std::isfinite(b.y2)
                &&b.x2>b.x1&&b.y2>b.y1, "ROI rectangle must be finite and nonempty");
        return b;
    }
    require(s.polygon.size()>=3, "A polygon needs at least three vertices");
    Box b{s.polygon[0].x,s.polygon[0].y,s.polygon[0].x,s.polygon[0].y};
    double area=0;
    for (size_t i=0;i<s.polygon.size();++i) {
        auto p=s.polygon[i], q=s.polygon[(i+1)%s.polygon.size()];
        require(std::isfinite(p.x)&&std::isfinite(p.y),"Polygon coordinates must be finite");
        b.x1=std::min(b.x1,p.x); b.x2=std::max(b.x2,p.x);
        b.y1=std::min(b.y1,p.y); b.y2=std::max(b.y2,p.y);
        area+=p.x*q.y-q.x*p.y;
    }
    require(std::abs(area)>1e-9&&b.x2>b.x1&&b.y2>b.y1,"Polygon has no area");
    return b;
}
inline View makeView(Box b,int iw,int ih,int ow,int oh,double margin) {
    require(iw>0&&ih>0&&ow>0&&oh>0,"Image sizes must be positive");
    require(std::isfinite(margin)&&margin>=0&&margin<=0.5,"Crop margin must be in [0, .5]");
    // Clip before adding context: an out-of-image ROI must not become a background crop.
    b={std::clamp(b.x1,0.0,double(iw)),std::clamp(b.y1,0.0,double(ih)),
       std::clamp(b.x2,0.0,double(iw)),std::clamp(b.y2,0.0,double(ih))};
    require(b.x2>b.x1&&b.y2>b.y1,"ROI does not intersect the image");
    double dx=(b.x2-b.x1)*margin,dy=(b.y2-b.y1)*margin;
    int x=int(std::floor(std::max(0.,b.x1-dx))),y=int(std::floor(std::max(0.,b.y1-dy)));
    int x2=int(std::ceil(std::min(double(iw),b.x2+dx))),y2=int(std::ceil(std::min(double(ih),b.y2+dy)));
    double scale=std::min(double(ow)/(x2-x),double(oh)/(y2-y));
    int rw=std::clamp(int(std::lround((x2-x)*scale)),1,ow);
    int rh=std::clamp(int(std::lround((y2-y)*scale)),1,oh);
    return {x,y,x2-x,y2-y,ow,oh,rw,rh,(ow-rw)/2,(oh-rh)/2,double(rw)/(x2-x),double(rh)/(y2-y)};
}
// First view is always the full ROI. Optional partitions have complete long-axis coverage.
// They are off by default, and can never turn one ROI into an unbounded number of forwards.
inline std::vector<View> planViews(const Shape& shape,int iw,int ih,int ow,int oh,
                                   int patch,double margin,int max_detail_views=0) {
    require(patch>0&&ow%patch==0&&oh%patch==0,"Input size must be a multiple of patch size");
    require(max_detail_views>=0&&max_detail_views<=3,"Detail view cap must be between 0 and 3");
    auto b=bounds(shape);
    std::vector<View> views{makeView(b,iw,ih,ow,oh,margin)};
    const auto& v=views.front();
    b={std::max(b.x1,0.),std::max(b.y1,0.),std::min(b.x2,double(iw)),std::min(b.y2,double(ih))};
    double short_tokens=std::min((b.x2-b.x1)*v.sx,(b.y2-b.y1)*v.sy)/patch;
    if (max_detail_views<2||short_tokens>=4) return views;
    int count=max_detail_views;
    bool horizontal=b.x2-b.x1>=b.y2-b.y1;
    for (int i=0;i<count;++i) {
        Box part=b;
        if (horizontal) { part.x1=b.x1+(b.x2-b.x1)*i/count; part.x2=b.x1+(b.x2-b.x1)*(i+1)/count; }
        else { part.y1=b.y1+(b.y2-b.y1)*i/count; part.y2=b.y1+(b.y2-b.y1)*(i+1)/count; }
        views.push_back(makeView(part,iw,ih,ow,oh,margin));
    }
    return views;
}
inline void addInterval(std::vector<float>& mask,int row,int width,double a,double b,float weight) {
    a=std::clamp(a,0.,double(width)); b=std::clamp(b,0.,double(width));
    if (b<=a) return;
    for (int x=int(std::floor(a));x<int(std::ceil(b));++x)
        mask[size_t(row)*width+x]+=float(std::max(0.,std::min(b,double(x+1))-std::max(a,double(x))))*weight;
}
// Rectangle coverage is exact. Simple concave polygons use scanlines with exact horizontal
// coverage and four vertical samples/pixel. No original-resolution mask allocation.
inline std::vector<float> rasterMask(const Shape& shape,const View& view,int samples=4) {
    require(samples>0,"Mask sampling must be positive");
    int w=view.output_width,h=view.output_height;
    std::vector<float> mask(size_t(w)*h,0);
    if (shape.polygon.empty()) {
        auto b=bounds(shape); auto p=view.map({b.x1,b.y1}),q=view.map({b.x2,b.y2});
        double y1=std::max(double(view.top),p.y),y2=std::min(double(view.top+view.resized_height),q.y);
        for (int y=std::max(0,int(std::floor(y1)));y<std::min(h,int(std::ceil(y2)));++y)
            addInterval(mask,y,w,std::max(double(view.left),p.x),std::min(double(view.left+view.resized_width),q.x),
                        float(std::max(0.,std::min(y2,double(y+1))-std::max(y1,double(y)))));
    } else {
        std::vector<Point> vertices; vertices.reserve(shape.polygon.size());
        for (auto p:shape.polygon) vertices.push_back(view.map(p));
        std::vector<double> xs; xs.reserve(vertices.size());
        for (int y=view.top;y<view.top+view.resized_height;++y) for (int sub=0;sub<samples;++sub) {
            double yy=y+(sub+.5)/samples; xs.clear();
            for (size_t i=0;i<vertices.size();++i) {
                auto a=vertices[i],b=vertices[(i+1)%vertices.size()];
                if ((a.y<=yy&&b.y>yy)||(b.y<=yy&&a.y>yy))
                    xs.push_back(a.x+(yy-a.y)*(b.x-a.x)/(b.y-a.y));
            }
            std::sort(xs.begin(),xs.end());
            for (size_t i=0;i+1<xs.size();i+=2)
                addInterval(mask,y,w,std::max(double(view.left),xs[i]),
                            std::min(double(view.left+view.resized_width),xs[i+1]),1.f/samples);
        }
    }
    return mask;
}
inline std::vector<float> patchWeights(const std::vector<float>& mask,int w,int h,int patch) {
    require(patch>0&&w%patch==0&&h%patch==0&&mask.size()==size_t(w)*h,"Mask/patch shape mismatch");
    int gw=w/patch;
    std::vector<float> weights(size_t(gw)*(h/patch),0.f);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x)
        weights[size_t(y/patch)*gw+x/patch]+=mask[size_t(y)*w+x]/(patch*patch);
    return weights;
}
inline double l2Normalize(float* values,size_t n) {
    double sq=0;
    for (size_t i=0;i<n;++i) { require(std::isfinite(values[i]),"Non-finite ROI feature"); sq+=double(values[i])*values[i]; }
    require(sq>1e-20,"ROI feature has zero norm; cannot index it");
    double norm=std::sqrt(sq);
    for (size_t i=0;i<n;++i) values[i]=float(values[i]/norm);
    return norm;
}
// Spatial information is opt-in for shape-sensitive data. The default has exactly D channels.
// Empty quadrants reuse global to keep branch norm/weight consistent for sparse polygons.
inline std::vector<float> pool(const float* tokens,int gh,int gw,int d,
                               const std::vector<float>& weights,double spatial_weight=0) {
    require(tokens&&gh>0&&gw>0&&d>0&&weights.size()==size_t(gh)*gw,"Token shape mismatch");
    require(std::isfinite(spatial_weight)&&spatial_weight>=0&&spatial_weight<1,"Spatial weight must be in [0,1)");
    int branches=spatial_weight>0?5:1;
    std::vector<double> sums(size_t(branches)*d,0),mass(branches,0);
    for (int y=0;y<gh;++y) for (int x=0;x<gw;++x) {
        size_t i=size_t(y)*gw+x; double a=weights[i];
        require(std::isfinite(a)&&a>=0,"Mask weights must be finite and non-negative");
        if (a==0) continue; // masked NaNs must never leak into the descriptor
        double norm=0;
        for(int c=0;c<d;++c) { float t=tokens[i*d+c];require(std::isfinite(t),"Non-finite foreground token");norm+=double(t)*t; }
        if (norm<=1e-20) continue;
        norm=std::sqrt(norm); mass[0]+=a;
        int q=1+(2*y>=gh?2:0)+(2*x>=gw?1:0);
        if(branches==5) mass[q]+=a;
        for(int c=0;c<d;++c) {
            double value=a*tokens[i*d+c]/norm;sums[c]+=value;
            if(branches==5)sums[size_t(q)*d+c]+=value;
        }
    }
    require(mass[0]>1e-8,"ROI mask has no usable patch support");
    std::vector<float> out(size_t(branches)*d);
    for(int b=0;b<branches;++b) {
        if(mass[b]>1e-8) {
            for(int c=0;c<d;++c)out[size_t(b)*d+c]=float(sums[size_t(b)*d+c]/mass[b]);
            double sq=0;for(int c=0;c<d;++c)sq+=double(out[size_t(b)*d+c])*out[size_t(b)*d+c];
            if(b>0&&sq<=1e-20) std::copy_n(out.data(),d,out.data()+size_t(b)*d);
            else l2Normalize(out.data()+size_t(b)*d,d);
        } else std::copy_n(out.data(),d,out.data()+size_t(b)*d);
    }
    if(branches==5) for(int b=0;b<5;++b) {
        float scale=float(std::sqrt(b==0?1-spatial_weight:spatial_weight/4));
        for(int c=0;c<d;++c)out[size_t(b)*d+c]*=scale;
    }
    return out;
}
inline std::vector<float> fuse(const std::vector<std::vector<float>>& views,
                               const std::vector<double>& source_areas,double detail_weight=.25) {
    require(!views.empty()&&views.size()==source_areas.size(),"View fusion shape mismatch");
    require(std::isfinite(detail_weight)&&detail_weight>=0&&detail_weight<1,"Detail weight must be in [0,1)");
    auto out=views.front();
    double mass=0;for(size_t i=1;i<views.size();++i)mass+=source_areas[i];
    if(views.size()>1&&mass>0) {
        for(float& f:out)f*=float(1-detail_weight);
        for(size_t i=1;i<views.size();++i) {
            require(views[i].size()==out.size(),"Descriptor dimensions differ across views");
            for(size_t j=0;j<out.size();++j)out[j]+=float(detail_weight*source_areas[i]/mass)*views[i][j];
        }
    }
    l2Normalize(out.data(),out.size());return out;
}
} // namespace irt::features::embedding
