#ifndef MQTT_H
#define MQTT_H

#include <mosquitto.h>
#include <opencv2/core/types.hpp>

class mqtt
{
    const char *mqtt_host="localhost";
    const char *mqtt_clientid="face-detecor";
    const char *mqtt_topic_prefix="video/0/facedetect";

public:
    mqtt();
    int connect();

    int publish_info_topic_point(const char *prefix, const char *topic, cv::Point2f p, float area, float conf);
    int publish_info_topic_int(const char *prefix, const char *topic, int value);
    ~mqtt();
private:
    struct mosquitto *m_mqtt=NULL;
};

#endif // MQTT_H
