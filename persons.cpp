
#include "persons.h"

using namespace std;

Persons::Persons(OpenFace *of, pqxx::connection *cx)
    : of(of), cx(cx)
{}

void Persons::save(cv::Mat vec, int faceid)
{
    std::string s;
    std::string e;

    e << vec;

    s="INSERT INTO faces (person, embedding) VALUES ("+std::to_string(faceid)+",'"+e+"');";

    pqxx::work t(*cx);
    t.exec(s);
    t.commit();
}

void Persons::save(cv::Mat vec)
{
    save(vec, current());
}

int Persons::query_closest(cv::Mat vec)
{
    std::string s;
    std::string e;
    int id=-1;

    e << vec;

    s="SELECT person FROM faces ORDER BY embedding <=> '"+e+"' LIMIT 1";

    cout << s << endl;

    pqxx::work t(*cx);

    auto res=t.exec(s);
    for (const auto &row : res) {
        id=row["person"].as<int>();
    }
    t.commit();

    return id;
}

int Persons::load_embeddings()
{
    std::string s;

    s="SELECT person,embedding AS e FROM faces ORDER BY person";

    pqxx::work t(*cx);

    auto res=t.exec(s);

    embeddings.clear();
    for (const auto &row : res) {
        std::vector<float> tmp;
        int id=row["person"].as<int>();
        std::string e=row["e"].as<std::string>();

        //cout << id << " = " << e << endl;

        // remove []
        std::stringstream se(e.substr(1, e.size()-2));

        // get the numbers x,y,x,,,,
        std::string t;
        while (std::getline(se, t, ',')) {
            //cout << t << endl;
            tmp.push_back(std::stof(t));
        }

        cv::Mat m(1, 128, CV_32F, tmp.data());

        of->store(m, id);

        m.copyTo(embeddings[id]);
    }

    t.commit();

    of->train();

    return embeddings.size();
}

int Persons::load_persons()
{
    std::string s;

    s="SELECT person,name FROM persons";

    pqxx::work t(*cx);

    auto res=t.exec(s);

    persons.clear();
    for (const auto &row : res) {
        int id=row["person"].as<int>();
        std::string n=row["name"].as<std::string>();

        cout << id << " = " << n << endl;

        persons[id]=n;
    }

    t.commit();

    cp=persons.begin();

    return persons.size();
}

int Persons::load()
{
    int r=load_persons();
    load_embeddings();

    return r;
}

int Persons::next() {
    if (cp!=persons.end())
        cp=std::next(cp);
    return cp->first;
}

int Persons::previous() {
    if (cp!=persons.begin())
        cp=std::prev(cp);
    return cp->first;
}

int Persons::current() {
    return cp->first;
}

int Persons::find(int id)
{
    if (cp = persons.find(id); cp != persons.end()) {
        return cp->first;
    }
    return -1;
}

cv::Mat Persons::current_embedding()
{
    return get_embedding(current());
}

std::string Persons::current_name() {
    return cp->second;
}

cv::Mat Persons::get_embedding(int id) const
{
    cv::Mat e(1, 128, CV_32F, cv::Scalar::all(0));

    if (auto search = embeddings.find(id); search != embeddings.end()) {
        cout << id << search->second << endl;

        e=search->second;
    }

    return e;
}

std::string Persons::get_name(int id) const
{
    if (auto search = persons.find(id); search != persons.end()) {
        cout << id << search->second << endl;
        return search->second;
    }

    return "";
}
