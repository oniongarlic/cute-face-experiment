#ifndef PERSONS_H
#define PERSONS_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include "openface.hpp"

#include <pqxx/pqxx>

class Persons
{
public:
    Persons();
    Persons(OpenFace *of, pqxx::connection *cx);
    void save(cv::Mat vec, int faceid);
    void save(cv::Mat vec);

    int load_embeddings();
    int load_persons();
    int load();

    int query_closest(cv::Mat vec);

    cv::Mat get_embedding(int id) const;
    std::string get_name(int id) const;

    int next();
    int previous();
    int current();
    int find(int id);
    std::string current_name();
    cv::Mat current_embedding();
    int find_person_id(const cv::Mat &e, float thres);
    bool is_same_person(int pid, const cv::Mat &e, float thres);
    bool is_same_embedding(const cv::Mat &e1, const cv::Mat &e2, float thres);
private:
    OpenFace *of;
    pqxx::connection *cx;
    std::multimap<int, cv::Mat> embeddings;
    std::map<int, std::string> persons;

    std::map<int, std::string>::iterator cp;
};

#endif // PERSONS_H
