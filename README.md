# A little face expirement in C++ and OpenCV

A face detection, recognition, embedding and pgvector experiment in C++, OpenCV and PostgreSQL with pgvector.

Needs a database with pgvector support and the following tables:

 CREATE EXTENSION vector;
 CREATE TABLE faces (person int, embedding vector(128));
 CREATE TABLE persons (person int primary key, name varchar(128));
 CREATE INDEX ON faces USING hnsw (embedding vector_l2_ops);

* YoloV8-Face code from https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn
