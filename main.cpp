#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>


using namespace std;
using namespace cv;

struct myStruct{
    cv::Mat img;
    cv::Mat descriptor;
};

void createVocabulary();    //创建字典

int main( int argc, char** argv ){
   
    /*生成字典的部分*/
     createVocabulary(); return 0;
    /*使用字典进行比对*/ 
    string path = "../vocabulary/vocabulary.yml.gz";
    DBoW3::Vocabulary vocabulary(path);
    if(vocabulary.empty()){
        cout<<"没有字典生成"<<endl;
        return 0;
    }
    
    int key;
    vector<KeyPoint> keypoints;
    vector<myStruct> v;
    Mat frame;
    VideoCapture capture(1);
    Ptr<Feature2D> detector = ORB::create();
    chrono::steady_clock::time_point start, end;
    chrono::duration<double> time_used;
    waitKey(1000);      //延时1秒让相机调整白平衡和曝光
    while(true){
        key = waitKey(50);  //每秒处理10帧
        start = chrono::steady_clock::now();
        if(key == 'q' || key == 27) 
            break;
        capture>>frame;
        frame = frame.colRange(0,frame.cols/2); //只取左目
        imshow("实时画面",frame);
        Mat descriptor;
        detector->detectAndCompute(frame,Mat(),keypoints,descriptor);
        // 如果v是空的，说明是第一张图片，直接继续
        if(v.empty()) {
            myStruct a;
            a.descriptor = descriptor;
            a.img = frame;
            v.push_back(a);
            continue;
        }
        // 把当前帧与上一帧相比较，如果相似度过高
        // 表明相机没有移动，不要存储描述子
        DBoW3::BowVector v1, v2;
        vocabulary.transform(descriptor,v1);
        vocabulary.transform(v[v.size()-1].descriptor,v2);
        double score = vocabulary.score(v1,v2);
        if(score > 0.08 ) continue;
        // 把描述子和图片存入向量
        myStruct a;
        a.descriptor = descriptor;
        a.img = frame;
        v.push_back(a);
        //从头开始进行相似度计算
        double last_score = 1;  //上一次对比的得分
        cout<<"begin  "<<v.size()<<endl;
        imshow("正在匹配的画面",a.img);
        vocabulary.transform(v[0].descriptor, v2);
        last_score = vocabulary.score(v1,v2);
        if(last_score < 0.02 ) last_score = 0.02;
        for(int i = 0; i < (int)v.size()-4; i++){        //v.size()-4是为了避免和相近的帧比较
            vocabulary.transform(v[i].descriptor, v2);
            score = vocabulary.score(v1,v2);
            cout<<score<<endl;
            imshow("第"+to_string(i)+"帧",v[i].img);
            if(score < 0.02) score = 0.02;      //相当于给score一个下限，防止score太低对后面造成误判
            //  当前分数超过上次分数的三倍认为检测到回环
            if(score > 0.15 ){
                cout<<"检测到回环"<<endl;
                cout<<"与第"<<i<<"帧相匹配"<<endl;
                vector<myStruct>::iterator it = v.begin();
                it += i;
                v.erase(it);      //删除与当前帧像匹配的之前的帧，避免vector变得太大
                waitKey(0);
                break;
            }
            if(score > 3*last_score || last_score > 3*score){
                cout<<"检测到回环"<<endl;
                //imshow("匹配的画面",v[i].img);
                cout<<"与第"<<i<<"帧相匹配"<<endl;
                vector<myStruct>::iterator it = v.begin();
                it += i;
                v.erase(it);      //删除与当前帧像匹配的之前的帧，避免vector变得太大
                waitKey(0);
                break;
            }
            last_score = score;
        }
        end = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(end-start);
        cout<<"每一帧用时"<<time_used.count()<<"ms"<<endl;
    }
    
    return 0;
}

void createVocabulary(){
     
    vector<Mat> images;
    vector<String> files;
    glob("../vocabulary/*.jpg",files);
    cout<<files.size()<<endl;
    for(int i = 0; i < files.size(); i++){
        images.push_back(imread(files[i]));
    }
    cout<<images.size()<<endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for(Mat& image:images){
        vector<KeyPoint> keypoint;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoint, descriptor);
        descriptors.push_back(descriptor);
    }

    DBoW3::Vocabulary vocabulary;
    vocabulary.create(descriptors);
    cout<<"vocabulary  info:"<<vocabulary<<endl;
    vocabulary.save("../vocabulary/vocabulary.yml.gz");
}
