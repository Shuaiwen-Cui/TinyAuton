/**
 * @file mqtt.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the function prototypes for mqtt connection.
 * @version 1.0
 * @date 2025-03-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __MQTT_H__
#define __MQTT_H__

/* Dependencies */
#include <stdio.h>
#include "esp_log.h"
#include "mqtt_client.h"

/* Macros */
#define MQTT_ADDRESS "mqtt://8.222.194.160" // MQTT Broker URL
#define MQTT_PORT 1883                      // MQTT Broker Port
#define MQTT_CLIENT "ESP32-S3-Node-001" // Client ID (Unique for devices)
#define MQTT_USERNAME "cshwstem"            // MQTT Username
#define MQTT_PASSWORD "Cshw0918#"           // MQTT Password

#define MQTT_PUBLIC_TOPIC      "/mqtt/node"       // publish topic
#define MQTT_SUBSCRIBE_TOPIC   "/mqtt/server"     // subscribe topic

/* Variables */
extern const char *TAG_MQTT; // tag for logging
extern esp_mqtt_client_handle_t s_mqtt_client;
extern bool s_is_mqtt_connected;

/* Function Prototypes */
/**
 * @brief MQTT client initialization and connection
 */
void mqtt_app_start(void);

#endif /* __MQTT_H__ */