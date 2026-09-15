#include "DinoRetrievalCore.hpp"
#include <chrono>
#include <iostream>
#include <random>
using namespace irt::features::priv;
using namespace irt::features::priv::retrieval;
int main(){
 const int dim=96,representatives=64,query_count=32,views=1000;
 std::mt19937 random(77);std::normal_distribution<float> normal;
 std::vector<float> q(query_count*dim),g(representatives*dim);
 for(auto* data:{&q,&g}){for(float& x:*data)x=normal(random);for(size_t i=0;i<data->size();i+=dim){std::vector<float> v(data->begin()+i,data->begin()+i+dim);normalize(v);std::copy(v.begin(),v.end(),data->begin()+i);}}
 std::vector<Evidence> evidence;std::vector<DinoPoint> points;
 for(int t=0;t<query_count;++t)evidence.push_back({{double(t%4)+.5,double(t/4)+.5},t/2,1.f/query_count});
 for(int i=0;i<representatives;++i)points.push_back({double(i%8)*32+16,double(i/8)*32+16});
 auto now=[](){return std::chrono::steady_clock::now();};
 auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
 float sink=0;auto start=now();
 for(int v=0;v<views;++v){auto pair=nearestTwo(g.data(),representatives,q.data(),query_count,dim);sink+=pair[0].first.score;}
 auto match_end=now();auto pairs=nearestTwo(g.data(),representatives,q.data(),query_count,dim);
 for(int v=0;v<views;++v){auto poses=vote(evidence,pairs,points,{0,0,4,8},32,32);if(!poses.empty())sink+=poses[0].score;}
 auto vote_end=now();std::vector<float> sim(query_count*1024);for(float& x:sim)x=normal(random)*.2f;
 SparseMatcher matcher(32,32,evidence,{0,0,4,8},sim,std::vector<float>(1024,1.f));
 FineOptions options;auto locate_start=now();auto located=matcher.locate(options);auto end=now();
 std::cout<<"measurement: synthetic_cpu_kernel_microbenchmark\nviews: "<<views<<"\nquery_tokens: "<<query_count
 <<"\nrepresentatives: "<<representatives<<"\ndimension: "<<dim<<"\nnearest_two_ms: "<<ms(start,match_end)
 <<"\nspatial_vote_ms: "<<ms(match_end,vote_end)<<"\nscalar_map_fine_one_crop_ms: "<<ms(locate_start,end)
 <<"\nend_to_end_dino_latency_ms: null\nconsume: "<<sink+located.front().score<<"\n";
}
