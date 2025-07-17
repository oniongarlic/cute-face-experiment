#include "mqtt.h"
#include <cstdio>
#include <cstring>


mqtt::mqtt()
{
    mosquitto_lib_init();
}

mqtt::~mqtt()
{
    mosquitto_destroy(m_mqtt);
    mosquitto_lib_cleanup();
}

static void mqtt_log_callback(struct mosquitto *m, void *userdata, int level, const char *str)
{
    // fprintf(stderr, "[MQTT-%d] %s\n", level, str);
}

int mqtt::connect(void)
{
    int port = 1883;
    int keepalive = 120;
    bool clean_session = true;

    m_mqtt=mosquitto_new(clientid, clean_session, NULL);
    mosquitto_log_callback_set(m_mqtt, mqtt_log_callback);

    printf("Connecting to MQTT...\n");

    if (mosquitto_connect(m_mqtt, host, port, keepalive)) {
        fprintf(stderr, "Unable to connect.\n");
        return -1;
    }
    return 0;
}

void mqtt::loop() {
    mosquitto_loop(m_mqtt, 1, 1);
}

int mqtt::publish_string(const char *topic, const char *data)
{
    int r;
    char ftopic[80];

    snprintf(ftopic, sizeof(ftopic), "%s/%s", prefix, topic);

    r=mosquitto_publish(m_mqtt, NULL, ftopic, strlen(data), data, 0, false);
    if (r!=MOSQ_ERR_SUCCESS)
        fprintf(stderr, "MQTT Publish for info [%s] failed with %s\n", topic, mosquitto_strerror(r));

    return r;
}

int mqtt::publish_point(const char *topic, cv::Point2f p, float area, float conf)
{
    int r;
    char ftopic[80];
    char data[256];

    snprintf(ftopic, sizeof(ftopic), "%s/%s", prefix, topic);
    snprintf(data, sizeof(data), "{\"face\": [%f,%f,%f,%f]}", p.x, p.y, area, conf);

    r=mosquitto_publish(m_mqtt, NULL, ftopic, strlen(data), data, 0, false);
    if (r!=MOSQ_ERR_SUCCESS)
        fprintf(stderr, "MQTT Publish for info [%s] failed with %s\n", topic, mosquitto_strerror(r));

    return r;
}

int mqtt::publish_int(const char *topic, int value)
{
    int r;
    char ftopic[80];
    char data[256];

    snprintf(ftopic, sizeof(ftopic), "%s/%s", prefix, topic);
    snprintf(data, sizeof(data), "%d", value);

    r=mosquitto_publish(m_mqtt, NULL, ftopic, strlen(data), data, 0, false);
    if (r!=MOSQ_ERR_SUCCESS)
        fprintf(stderr, "MQTT Publish for info [%s] failed with %s\n", topic, mosquitto_strerror(r));

    return r;
}
