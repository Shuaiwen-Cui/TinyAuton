/**
 * @file node_wifi.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the functions to connect to a WPA2 Enterprise network.
 * @version 1.0
 * @date 2025-02-24
 *
 * @copyright Copyright (c) 2025
 *
 */

/* Dependencies */
#include "node_wifi.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* Variables */
    const char *TAG_WIFI = "NODE-WIFI";  // tag for logging
    EventGroupHandle_t wifi_event_group; // FreeRTOS event group to signal when we are connected & ready to make a request
    esp_netif_t *sta_netif = NULL;       // esp netif object representing the WIFI station
    const int CONNECTED_BIT = BIT0;      // use bit 0 to indicate whether connected

    /**
     * @name wifi_event_handler
     * @brief Event handler for WIFI and IP events
     * @param arg argument passed to the handler
     * @param event_base base of the event
     * @param event_id id of the event
     * @param event_data data of the event
     * @return void
     *
     */
    void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
    {
        if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
        {
            esp_wifi_connect();
        }
        else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
        {
            esp_wifi_connect();
            xEventGroupClearBits(wifi_event_group, CONNECTED_BIT); // clear the connected bit
        }
        else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
        {
            xEventGroupSetBits(wifi_event_group, CONNECTED_BIT); // only when the ip is obtained, set the connected bit
        }
    }

    /**
     * @name wifi_sta_wpa2_init
     * @brief Initialize the WIFI station to connect to a WPA2 Enterprise network
     * @param void
     * @return esp_err_t
     */
    esp_err_t wifi_sta_wpa2_init(void)
    {
        esp_err_t ret;

        /* Preparation */
        ESP_ERROR_CHECK(esp_netif_init());                // TCP/IP stack initialization
        wifi_event_group = xEventGroupCreate();           // create the event group
        ESP_ERROR_CHECK(esp_event_loop_create_default()); // create the event group, later can be used to deal with a series of events by registing call back functions
        sta_netif = esp_netif_create_default_wifi_sta();  // creat STA object using default configuration
        assert(sta_netif);

        /* WiFi Configuration*/
        wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();                                                   // use default configuration
        ESP_ERROR_CHECK(esp_wifi_init(&cfg));                                                                  // initialize the wifi using the configuration struct
        ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));  // register the event handler for WIFI events: port wifi events to the wifi_event_handler for processing
        ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL)); // register the event handler for IP events: port ip events to the wifi_event_handler for processing
        ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));                                               // set the storage to RAM
        wifi_config_t wifi_config = {
            .sta = {
                .ssid = ENTERPRISE_WIFI_SSID, // set the SSID
            },
        };
        ESP_LOGI(TAG_WIFI, "Setting WiFi configuration SSID %s...", wifi_config.sta.ssid);
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));                   // set the mode to station mode
        ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config)); // set the configuration with the configured struct
        // ESP_ERROR_CHECK( esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)EXAMPLE_EAP_ID, strlen(EXAMPLE_EAP_ID)) ); // set the identity (not necessary for WAP2-ENTERPRISE)

        ESP_ERROR_CHECK(esp_eap_client_set_username((uint8_t *)ENTERPRISE_WIFI_USERNAME, strlen(ENTERPRISE_WIFI_USERNAME))); // ESP_ERROR_CHECK(esp_wifi_sta_wpa2_ent_set_username((uint8_t *)ENTERPRISE_WIFI_USERNAME, strlen(ENTERPRISE_WIFI_USERNAME)));
        ESP_ERROR_CHECK(esp_eap_client_set_password((uint8_t *)ENTERPRISE_WIFI_PASSWORD, strlen(ENTERPRISE_WIFI_PASSWORD))); // ESP_ERROR_CHECK(esp_wifi_sta_wpa2_ent_set_password((uint8_t *)ENTERPRISE_WIFI_PASSWORD, strlen(ENTERPRISE_WIFI_PASSWORD)));

        ESP_ERROR_CHECK(esp_wifi_sta_enterprise_enable()); // ESP_ERROR_CHECK(esp_wifi_sta_wpa2_ent_enable());

        /* WiFi Start */
        ret = esp_wifi_start();
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG_WIFI, "esp_wifi_start failed: %s", esp_err_to_name(ret));
            return ret;
        }
        return ESP_OK;
    }

#ifdef __cplusplus
}
#endif
