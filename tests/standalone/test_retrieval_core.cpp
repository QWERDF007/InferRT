#include "DinoRetrievalCore.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <set>
using namespace irt::features::priv;
using namespace irt::features::priv::retrieval;
namespace {
int checks = 0;
void require(bool x, const char* message) {++checks;if(!x)throw std::runtime_error(message);}
void close(double a,double b,double eps,const char* message){require(std::abs(a-b)<=eps,message);}
DinoFeatureGrid grid(int h,int w,int d=8,int patch=1) {
    DinoFeatureGrid g;g.channels=d;g.plan.grid_height=h;g.plan.grid_width=w;
    g.plan.patch_size=patch;g.plan.input_width=w*patch;g.plan.input_height=h*patch;
    g.plan.valid_input_rect={0,0,double(w*patch),double(h*patch)};
    g.plan.source_rect=g.plan.valid_input_rect;
    g.tokens.assign(size_t(h*w*d),0.f);g.valid_area.assign(size_t(h*w),1.f);
    for(int i=0;i<h*w;++i)g.tokens[size_t(i*d+(i%d))]=1.f;
    return g;
}
void projectionTests() {
    std::mt19937 rng(13);std::normal_distribution<float> random;
    for(int dim:{96,192,384}){
        Projection p(384,dim),again(384,dim);
        for(int n=0;n<12;++n){
            std::vector<float> a(384);for(auto& x:a)x=random(rng);normalize(a);
            auto b=p.apply(a.data());require(b==again.apply(a.data()),"projection deterministic");
            close(dot(b.data(),b.data(),dim),1.,2e-6,"unit projected vector");
            if(dim==384)require(a==b,"full dimension identity");
            float maximum=0;for(float x:b)maximum=std::max(maximum,std::abs(x));
            std::vector<float> decoded;for(float x:b)decoded.push_back(std::nearbyint(x/maximum*127.f));normalize(decoded);
            require(dot(b.data(),decoded.data(),dim)>.9998,"int8 angular error on random vectors");
        }
    }
    bool rejected=false;try{Projection invalid(384,500);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"invalid dimensions rejected");
}
void selectionTests() {
    auto g=grid(32,32,16,16);
    auto selected=select(g,{0,0,512,512},g.valid_area,4,4);
    require(selected.size()==64,"bounded 64 gallery representatives");
    std::set<int> ids;float sum=0;for(auto s:selected){ids.insert(s.patch);sum+=s.weight;}
    require(ids.size()==selected.size(),"no duplicate patches");close(sum,1.,1e-6,"normalized sample mass");
    require(regionWindows(g,0,.5f).size()==10,"ten full-grid region windows");
    auto wide=grid(8,32);auto wr=regionWindows(wide,0,.5f);
    require(wr.size()==10,"wide view bounded windows");require(wr[0].grid_width==32&&wr[0].grid_height==8,"whole wide region covered");
    auto one=grid(1,1);require(regionWindows(one,0,.5f).size()==1,"dedup tiny windows");
    DinoRect thin{250.4,70.2,261.44,440.8};std::vector<float> weights(1024,0);
    int old_centres=0;
    for(int r=0;r<32;++r)for(int c=0;c<32;++c){auto b=g.plan.patchRect(r,c);weights[size_t(r*32+c)]=float(DinoRect::intersect(thin,b).area()/b.area());
        if(thin.contains((b.x0+b.x1)*.5,(b.y0+b.y1)*.5))++old_centres;}
    require(old_centres==0,"thin regression has no patch centres");
    auto slim=select(g,thin,weights,4,2);require(!slim.empty(),"thin ROI keeps overlap evidence");
    sum=0;ids.clear();for(auto s:slim){sum+=s.weight;ids.insert(s.patch);require(thin.contains(s.point.x,s.point.y),"sample coordinate inside ROI");}
    close(sum,1,1e-6,"thin ROI mass preserved");require(ids.size()==slim.size(),"thin ROI no fake duplicated cells");
    weights.assign(1024,0);weights[0]=.01f;auto tiny=select(g,{1,1,2,2},weights,4,2);
    require(tiny.size()==1,"subpatch query yields one independent sample");close(tiny[0].weight,1,1e-6,"single representative keeps full weight");
    weights.assign(1024,0);require(select(g,{0,0,50,50},weights,4,2).empty(),"empty polygon mask no evidence");
    auto full=select(g,{0,0,512,512},g.valid_area,4,8);require(full.size()==128,"128 fallback cap");
    g.valid_area.assign(1024,0);require(regionWindows(g,0,.5f).empty(),"padding-only view empty");
}
void nearestTests() {
    const float gallery[]{1,0, 1,0, 0,1, -1,0};const float query[]{1,0,0,1};
    auto a=nearestTwo(gallery,4,query,2,2);
    require(a[0].first.index==0&&a[0].second.index==1,"ties stable, distinct indices");
    require(a[1].first.index==2,"different local appearance found");
    auto one=nearestTwo(gallery,1,query,2,2);require(one[0].second.index==-1,"singleton second invalid");
    Pair p;insert(p,{4,.5f});insert(p,{4,.7f});insert(p,{2,.6f});
    require(p.first.index==4&&p.second.index==2,"duplicate match not repeated");
}
void votingTests() {
    std::vector<Evidence> q{{{.25,.25},0,.25f},{{.75,.25},1,.25f},{{.25,.75},2,.25f},{{.75,.75},3,.25f}};
    for(double scale:{.25,.5,1.,2.,4.}) {
        std::vector<DinoPoint> positions;std::vector<Pair> pairs;
        for(size_t i=0;i<q.size();++i){positions.push_back({10+q[i].point.x*scale,20+q[i].point.y*scale});Pair p;p.first={int(i),.9f};pairs.push_back(p);}
        auto poses=vote(q,pairs,positions,{0,0,1,1},scale,.2*scale);
        require(!poses.empty(),"spatial pose found");close(poses[0].support,1,1e-6,"all independent cells vote");
        close(poses[0].box.x0,10,1e-6,"translation recovered");close(poses[0].box.width(),scale,1e-6,"scale recovered");
        auto scattered=positions;scattered[0].x-=5;scattered[1].y+=7;scattered[2].x+=9;
        auto wrong=vote(q,pairs,scattered,{0,0,1,1},scale,.2*scale);
        require(poses[0].score>wrong[0].score,"same appearances with wrong layout rank lower");
        auto duplicated=q;auto duplicated_pairs=pairs;
        for(auto& e:duplicated)e.weight*=.5f;
        for(size_t i=0;i<q.size();++i){auto e=q[i];e.weight*=.5f;duplicated.push_back(e);duplicated_pairs.push_back(pairs[i]);}
        auto d=vote(duplicated,duplicated_pairs,positions,{0,0,1,1},scale,.2*scale);
        close(d[0].support,1,1e-6,"duplicate token does not double cell support");
        close(d[0].score,poses[0].score,1e-6,"duplicate token cannot boost pose score");
    }
}
std::vector<float> mapsFor(const std::vector<Evidence>& q,int h,int w,const DinoRect& source,const DinoRect& target) {
    std::vector<float> sim(q.size()*size_t(h*w),-.8f);
    for(size_t t=0;t<q.size();++t){int x=int(target.x0+(q[t].point.x-source.x0)/source.width()*target.width());
        int y=int(target.y0+(q[t].point.y-source.y0)/source.height()*target.height());
        sim[t*size_t(h*w)+size_t(y*w+x)]=1.f;}
    return sim;
}
void fineTests() {
    // Continuous ROI coordinates with normal, wide and tall shapes; no patch-support bounding box.
    for(auto source:{DinoRect{0,0,2,2},DinoRect{10,20,14,22},DinoRect{7,9,8,13}}) {
        std::vector<Evidence> q;
        for(int y=0;y<2;++y)for(int x=0;x<2;++x)q.push_back({{source.x0+(.125+.75*x)*source.width(),source.y0+(.125+.75*y)*source.height()},y*2+x,.25f});
        DinoRect target{2,3,6,7};auto sim=mapsFor(q,12,12,source,target);
        SparseMatcher matcher(12,12,q,source,sim,std::vector<float>(144,1.f));FineOptions options;
        auto exact=matcher.evaluate(target,options);require(exact.score>.55f,"correct layout has evidence");
        auto wrong=matcher.evaluate({7,7,11,11},options);require(exact.score>wrong.score,"wrong placement lower");
        auto peaks=matcher.locate(options,target);require(!peaks.empty(),"fine returns candidate");
        require(iou(peaks.front().box,target)>.45,"fine localization normal/aspect boxes");
        auto zero=std::vector<float>(144,0.f);SparseMatcher missing(12,12,q,source,sim,zero);
        close(missing.evaluate(target,options).score,0,1e-8,"padding adds no score");
        close(missing.evaluate(target,options).coverage,0,1e-8,"missing evidence not renormalized away");
    }
    std::vector<Evidence> q{{{.5,.5},0,1.f}};FineOptions options;options.peaks=4;
    std::vector<float> sim(64,-1.f);sim[9]=1;sim[54]=1;
    SparseMatcher repeated(8,8,q,{0,0,1,1},sim,std::vector<float>(64,1));
    auto peaks=repeated.locate(options);bool first=false,second=false;
    for(auto p:peaks){first|=p.box.contains(1.5,1.5);second|=p.box.contains(6.5,6.5);}
    require(first&&second,"two spatial instances survive");
    require(repeated.evaluate({-.1,0,1,1},options).score<0,"reject out-of-grid trial");
    auto stopped=repeated.locate(options,{},[]{return true;});require(stopped.empty(),"deadline stops scan");
    // Composite score is evaluated before retention: lower appearance may still win.
    std::vector<float> scalar{.6f,.3f,-1.f,-1.f};SparseMatcher compound(1,4,q,{0,0,1,1},scalar,std::vector<float>(4,1.f));
    options.appearance_weight=0;options.coverage_weight=0;options.consistency_weight=1;
    require(compound.evaluate({0,0,1,1},options).score>compound.evaluate({1,0,2,1},options).score,"layout contribution affects full score");
}
}
int main(){try{projectionTests();selectionTests();nearestTests();votingTests();fineTests();
 std::cout<<"status: PASS\nchecks: "<<checks<<"\nscope: production_header_algorithms\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL after "<<checks<<" checks: "<<e.what()<<"\n";return 1;}}
