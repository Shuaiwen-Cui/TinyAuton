/**
 * @file mqtt.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the function prototypes for mqtt connection.
 * @version 1.0
 * @date 2025-03-17
 *
 * @copyright Copyright (c) 2025
 *
 */

/* Dependencies */
#include "mqtt.h"

/* Macros */

/* Variables */
const char *TAG_MQTT = "NODE-MQTT";             // tag for logging
esp_mqtt_client_handle_t s_mqtt_client = NULL; // MQTT client handle
bool s_is_mqtt_connected = false;              // MQTT connection status flag

/* Function Prototypes */

/**
 * @brief MQTT event handler
 *
 * @param event_handler_arg Argument passed to the event handler
 * @param event_base Event base identifier
 * @param event_id Event identifier
 * @param event_data Event-specific data
 */
void aliot_mqtt_event_handler(void *event_handler_arg,
                                     esp_event_base_t event_base,
                                     int32_t event_id,
                                     void *event_data)
{
    esp_mqtt_event_handle_t event = event_data;
    esp_mqtt_client_handle_t client = event->client;

    switch ((esp_mqtt_event_id_t)event_id)
    {
    case MQTT_EVENT_CONNECTED: // Connection established
        ESP_LOGI(TAG_MQTT, "MQTT connected");
        // Subscribe to the test topic upon successful connection
        esp_mqtt_client_subscribe_single(client, MQTT_SUBSCRIBE_TOPIC, 1);
        break;
    case MQTT_EVENT_DISCONNECTED: // Connection disconnected
        ESP_LOGI(TAG_MQTT, "MQTT disconnected");
        break;
    case MQTT_EVENT_SUBSCRIBED: // Subscription successful
        ESP_LOGI(TAG_MQTT, "MQTT subscribed, msg_id=%d", event->msg_id);
        break;
    case MQTT_EVENT_UNSUBSCRIBED: // Unsubscription successful
        ESP_LOGI(TAG_MQTT, "MQTT unsubscribed, msg_id=%d", event->msg_id);
        break;
    case MQTT_EVENT_PUBLISHED: // Publish acknowledgment received
        ESP_LOGI(TAG_MQTT, "MQTT published ack, msg_id=%d", event->msg_id);
        break;
    case MQTT_EVENT_DATA: // Data received
        printf("TOPIC=%.*s\r\n", event->topic_len, event->topic);
        printf("DATA=%.*s\r\n", event->data_len, event->data);
        break;
    case MQTT_EVENT_ERROR: // Error event
        ESP_LOGI(TAG_MQTT, "MQTT event error");
        break;
    default:
        ESP_LOGI(TAG_MQTT, "Unhandled MQTT event id: %ld", event_id);
        break;
    }
}

/**
 * @brief MQTT client initialization and connection
 */
void mqtt_app_start(void)
{
    esp_err_t ret;

    esp_mqtt_client_config_t mqtt_cfg = {0};
    mqtt_cfg.broker.address.uri = MQTT_ADDRESS;
    mqtt_cfg.broker.address.port = MQTT_PORT;

    // Client ID
    mqtt_cfg.credentials.client_id = MQTT_CLIENT;

    // Username
    mqtt_cfg.credentials.username = MQTT_USERNAME;

    // Password
    mqtt_cfg.credentials.authentication.password = MQTT_PASSWORD;

    ESP_LOGI(TAG_MQTT, "Connecting to MQTT broker...");

    // Initialize MQTT client with provided configuration
    esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);

    // Register MQTT event handler
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, aliot_mqtt_event_handler, client);

    // Start MQTT client
    ret = esp_mqtt_client_start(client);
    if(ret == ESP_OK)
    {
        ESP_LOGI(TAG_MQTT, "MQTT client started");
        s_mqtt_client = client;
        s_is_mqtt_connected = true;
    }
    else
    {
        ESP_LOGE(TAG_MQTT, "MQTT client start failed");
    }
}
