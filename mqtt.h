#ifndef MQTT_H
#define MQTT_H

#include <mosquitto.h>
#include <opencv2/core/types.hpp>

class mqtt
{
    const char *host="localhost";
    const char *clientid="face-detecor";
    const char *prefix="video/0/facedetect";

public:
    mqtt();
    ~mqtt();

    int connect();

    int publish_point(const char *topic, cv::Point2f p, float area, float conf);
    int publish_int(const char *topic, int value);
    int publish_string(const char *topic, const char *data);
private:
    struct mosquitto *m_mqtt=NULL;
};

#endif // MQTT_H
