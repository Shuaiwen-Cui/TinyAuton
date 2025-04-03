/**
 * @file measurement.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file is the source file for the measurement module.
 * @version 1.0
 * @date 2025-04-01
 * @copyright Copyright (c) 2025
 * @note Currently, the sesning is realized by FreeRTOS, and therefore the upper limit sampling rate is 1kHz.
 * @todo Sampling by timer + interrupt
 * @todo Sampling by DMA
 */

/* DEPENDENCIES */
#include "measurement.h"

/* VARIABLES */
static const char *TAG = "TinySHM-Measurement"; // Tag for logging

// static uint8_t mpu6050_deviceid;    // MPU6050 Device ID
static mpu6050_acce_value_t acce; // MPU6050 Accelerometer Value
// static mpu6050_gyro_value_t gyro;   // MPU6050 Gyroscope Value
static mpu6050_temp_value_t temp; // MPU6050 Temperature Value
// static complimentary_angle_t angle; // Complimentary Angle

/* DEFINITION */
struct SenseConfig
{
    int sample_rate;
    int sample_duration;
    bool temperature_sense; // whether to sense temperature
    bool printout;          // not recommened for high sample rate as it will slow down the process
    bool mqtt_pub;          // not recommened for high sample rate as it will slow down the process
};

struct DataStruct
{
    int sample_rate;
    int sample_duration;
    float sensor_temperature;
    TinyTimeMark_t start_time_stamp;
    TinyTimeMark_t *time_stamp;
    float *acc_data;
    float *temperature;
};

/* VARIABLES */
// for streamline configuration
struct SenseConfig streamline_config = {
    .sample_rate = 1,          // Sample rate in Hz. For local printout + mqtt, at most 25Hz; for only mqtt, at most 200Hz.
    .temperature_sense = true, // Whether to sense temperature
    .printout = true,          // Only available for sampling rate <= 100Hz
    .mqtt_pub = true           // Only available for sampling rate <= 100Hz
};

// for sensing configuration
struct SenseConfig sense_config = {
    .sample_rate = 500,   // Sample rate in Hz, only can be (1, 2, 5, 10, 20, 25, 50, 100, 200, 500) for FreeRTOS based sensing. Note can not be 1000Hz (no time to feed the watchdog).
    .sample_duration = 1, // Sample duration in seconds
    .printout = true      // Only available for sampling rate <= 100Hz
};

// for data structure
struct DataStruct data_struct = {
    .sample_rate = 0,     // Sample rate in Hz
    .sample_duration = 0, // Sample duration in seconds
    .sensor_temperature = 0.0,
    .start_time_stamp = 0,
    .acc_data = NULL};

/* FUNCTION PROTOTYPES */
/**
 * @name acc_streamline_task
 * @brief This function is the task for streaming the accelerometer data.
 * @param pvParameters for FreeRTOS, the passed parameters are all void pointers and need to be casted to the correct type.
 * @retval None
 */
void acc_streamline_task(void *pvParameters)
{

    // streamline variables
    float acc_x, acc_y, acc_z, temperature;
    TinyTimeMark_t time_stamp;

    // MQTT publish buffer
    char streamline_buff[256];

    struct SenseConfig *config = (struct SenseConfig *)pvParameters;
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1000 / config->sample_rate);

    if (config->sample_rate > 100 && config->printout)
    {
        ESP_LOGW(TAG, "Printout is disabled for sample rate > 100Hz.");
        config->printout = false;
    }

    while (1)
    {
        // get the time stamp
        time_stamp = tiny_get_time();

        // get accelerometer data
        mpu6050_get_acce(mpu6050, &acce);
        acc_x = acce.acce_x;
        acc_y = acce.acce_y;
        acc_z = acce.acce_z;

        // get temperature
        mpu6050_get_temp(mpu6050, &temp);
        temperature = temp.temp;

        // sprintf
        if (config->printout || config->mqtt_pub)
        {
            snprintf(streamline_buff, sizeof(streamline_buff), "Time Stamp: %10ld, acc_x = %10.6f, acc_y = %10.6f, acc_z = %10.6f, Sensor Temperature: %4.2f °C", time_stamp, acc_x, acc_y, acc_z, temperature);
        }

        // // print out the data
        if (config->printout)
        {
            ESP_LOGI(TAG, "%s", streamline_buff);
        }

        // MQTT publish
        if (config->mqtt_pub)
        {
            esp_mqtt_client_publish(s_mqtt_client, MQTT_PUBLIC_TOPIC, streamline_buff, strlen(streamline_buff), 0, 0); // for streamline, data quality is not expected to be high, so QoS = 0; data retaintion is also not required, so 0.
        }

        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
    ESP_LOGI(TAG, "Streamline terminated. Task will delete itself.");
    vTaskDelete(NULL);
}

/**
 * @name acc_sense_task
 * @brief This function is the task for sensing the accelerometer data according to the configuration structure.
 * @param pvParameters for FreeRTOS, the passed parameters are all void pointers and need to be casted to the correct type.
 * @retval None
 */
void acc_sense_task(void *pvParameters)
{
    struct SenseConfig *config = (struct SenseConfig *)pvParameters;
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1000 / config->sample_rate);
    int total_samples = config->sample_rate * config->sample_duration;

    // Fill global data_struct
    data_struct.sample_rate = config->sample_rate;
    data_struct.sample_duration = config->sample_duration;
    mpu6050_get_temp(mpu6050, &temp);
    data_struct.sensor_temperature = temp.temp;
    data_struct.start_time_stamp = tiny_get_time();

    // Allocate memory
    data_struct.acc_data = malloc(sizeof(float) * 3 * total_samples);
    data_struct.temperature = malloc(sizeof(float) * total_samples);
    data_struct.time_stamp = malloc(sizeof(TinyTimeMark_t) * total_samples);

    if (data_struct.acc_data == NULL || data_struct.temperature == NULL || data_struct.time_stamp == NULL)
    {
        ESP_LOGE(TAG, "Failed to allocate memory for sensing data.");
        if (data_struct.acc_data)
            free(data_struct.acc_data);
        if (data_struct.temperature)
            free(data_struct.temperature);
        if (data_struct.time_stamp)
            free(data_struct.time_stamp);
        vTaskDelete(NULL);
    }

    ESP_LOGI(TAG, "Sampling started: %d Hz for %d seconds (%d samples)",
             config->sample_rate, config->sample_duration, total_samples);

    for (int i = 0; i < total_samples; i++)
    {
        TinyTimeMark_t now = tiny_get_time();
        mpu6050_get_acce(mpu6050, &acce);
        mpu6050_get_temp(mpu6050, &temp);

        // Store data
        data_struct.time_stamp[i] = now;
        data_struct.acc_data[i * 3 + 0] = acce.acce_x;
        data_struct.acc_data[i * 3 + 1] = acce.acce_y;
        data_struct.acc_data[i * 3 + 2] = acce.acce_z;
        data_struct.temperature[i] = temp.temp;

        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }

    ESP_LOGI(TAG, "Sampling complete.");

    // Print metadata
    ESP_LOGI(TAG, "-----------------------------");
    ESP_LOGI(TAG, "Sensor Temperature : %.2f °C", data_struct.sensor_temperature);
    ESP_LOGI(TAG, "Start Time Stamp   : %ld", (long)data_struct.start_time_stamp);
    ESP_LOGI(TAG, "Sample Rate        : %d Hz", data_struct.sample_rate);
    ESP_LOGI(TAG, "Sample Duration    : %d sec", data_struct.sample_duration);
    ESP_LOGI(TAG, "Total Samples      : %d", total_samples);
    ESP_LOGI(TAG, "Data Size          : %.2f KB",
             (sizeof(float) * 4 + sizeof(TinyTimeMark_t)) * total_samples / 1024.0f);
    ESP_LOGI(TAG, "-----------------------------");

    // Print collected data if enabled
    if (config->printout)
    {
        for (int i = 0; i < total_samples; i++)
        {
            ESP_LOGI(TAG, "Time Stamp: %10ld, acc_x = %10.6f, acc_y = %10.6f, acc_z = %10.6f, Temperature: %.2f °C",
                     (long)data_struct.time_stamp[i],
                     data_struct.acc_data[i * 3 + 0],
                     data_struct.acc_data[i * 3 + 1],
                     data_struct.acc_data[i * 3 + 2],
                     data_struct.temperature[i]);
        }
    }

    // Free allocated memory
    if (data_struct.acc_data)
    {
        free(data_struct.acc_data);
        data_struct.acc_data = NULL;
    }
    if (data_struct.temperature)
    {
        free(data_struct.temperature);
        data_struct.temperature = NULL;
    }
    if (data_struct.time_stamp)
    {
        free(data_struct.time_stamp);
        data_struct.time_stamp = NULL;
    }

    vTaskDelete(NULL);
}


// void acc_sense_task(void *pvParameters)
// {
//     struct SenseConfig *config = (struct SenseConfig *)pvParameters;
//     TickType_t xLastWakeTime = xTaskGetTickCount();
//     const TickType_t xFrequency = pdMS_TO_TICKS(1000 / config->sample_rate);
//     int total_samples = config->sample_rate * config->sample_duration;

//     // Fill global data_struct
//     data_struct.sample_rate = config->sample_rate;
//     data_struct.sample_duration = config->sample_duration;
//     mpu6050_get_temp(mpu6050, &temp);
//     data_struct.sensor_temperature = temp.temp;
//     data_struct.start_time_stamp = tiny_get_time();
//     data_struct.acc_data = malloc(sizeof(float) * 3 * total_samples);
//     if (data_struct.acc_data == NULL)
//     {
//         ESP_LOGE(TAG, "Failed to allocate memory for acc_data.");
//         vTaskDelete(NULL);
//     }

//     ESP_LOGI(TAG, "Sampling started: %d Hz for %d seconds (%d samples)",
//              config->sample_rate, config->sample_duration, total_samples);

//     for (int i = 0; i < total_samples; i++)
//     {
//         mpu6050_get_acce(mpu6050, &acce);

//         // Save accelerometer data to global buffer
//         data_struct.acc_data[i * 3 + 0] = acce.acce_x;
//         data_struct.acc_data[i * 3 + 1] = acce.acce_y;
//         data_struct.acc_data[i * 3 + 2] = acce.acce_z;

//         // Print first few samples for debug
//         if (i < 5)
//         {
//             ESP_LOGI(TAG, "[DEBUG] Sample #%3d: x=%.3f, y=%.3f, z=%.3f",
//                      i + 1, acce.acce_x, acce.acce_y, acce.acce_z);
//         }

//         vTaskDelayUntil(&xLastWakeTime, xFrequency);
//     }

//     ESP_LOGI(TAG, "Sampling complete.");

//     // Print metadata
//     ESP_LOGI(TAG, "-----------------------------");
//     ESP_LOGI(TAG, "Sensor Temperature : %.2f °C", data_struct.sensor_temperature);
//     ESP_LOGI(TAG, "Start Time Stamp   : %d", data_struct.start_time_stamp);
//     ESP_LOGI(TAG, "Sample Rate        : %d Hz", data_struct.sample_rate);
//     ESP_LOGI(TAG, "Sample Duration    : %d sec", data_struct.sample_duration);
//     ESP_LOGI(TAG, "Total Samples      : %d", total_samples);
//     ESP_LOGI(TAG, "Data Size          : %d bytes", sizeof(float) * 3 * total_samples);
//     ESP_LOGI(TAG, "-----------------------------");

//     // Print collected data if enabled
//     if (config->printout)
//     {
//         for (int i = 0; i < total_samples; i++)
//         {
//             ESP_LOGI(TAG, "Sample #%3d: x=%.6f, y=%.6f, z=%.6f",
//                      i + 1,
//                      data_struct.acc_data[i * 3 + 0],
//                      data_struct.acc_data[i * 3 + 1],
//                      data_struct.acc_data[i * 3 + 2]);
//         }
//     }

//     // Free allocated memory if not needed further
//     if (data_struct.acc_data != NULL)
//     {
//         free(data_struct.acc_data);
//         data_struct.acc_data = NULL;
//     }

//     vTaskDelete(NULL);
// }
