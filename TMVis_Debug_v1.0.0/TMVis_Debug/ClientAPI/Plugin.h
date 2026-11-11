#ifndef _PLUGIN_H_
#define _PLUGIN_H_

#include <thread>
#include <iostream>
#include <mutex>
#include <functional>

#include "tcp_client.h"
#include "MessageBus.h"
#include "common/DataModel.h"
#include "common/StringParsers.h"

class Plugin {
public:
    Plugin(const std::string &name)
    {
        (void)name;
        _client = new TCPClient;
        _client->setProcessCallback(std::bind(&Plugin::process, this, std::placeholders::_1, std::placeholders::_2));
    }

    Plugin()
    {
        _client = new TCPClient;
        _client->setProcessCallback(std::bind(&Plugin::process, this, std::placeholders::_1, std::placeholders::_2));
    }

    virtual ~Plugin()
    {
        delete _client;
        _client = nullptr;
    }

    bool getSubscriberData(tmv::DepthFrame& out)
    {
        std::lock_guard<std::mutex> lock(_mtxDepth);
        out = _depthFrame;
        return out.valid;
    }

    bool getSubscriberData(tmv::ImageFrame& out)
    {
        std::lock_guard<std::mutex> lock(_mtxRgb);
        out = _rgbFrame;
        return out.valid;
    }

    bool getSubscriberRGBD(tmv::ImageFrame& out)
    {
        std::lock_guard<std::mutex> lock(_mtxRgbd);
        out = _rgbdFrame;
        return out.valid;
    }

    bool getSubscriberData(tmv::MarkerDetections& out)
    {
        std::lock_guard<std::mutex> lock(_mtxMarkers);
        out = _markerDetections;
        return out.valid;
    }

    bool getSubscriberData(tmv::SafetyStatus& out)
    {
        std::lock_guard<std::mutex> lock(_mtxSafety);
        out = _safetyStatus;
        return out.valid;
    }

    virtual void run() = 0;

    void start() {
        if (!_client) {
            std::cerr << "Client is null!\n";
            return;
        }

        _client->start();
        std::cout << "Waiting for load data...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));

        std::thread request_thread([this] {
            // Fire all 5 requests back-to-back, then sleep for the remainder
            // of a 33 ms period (≈ 30 Hz).
            //
            // Old pattern: 5 × 33 ms sequential sleep = 165 ms/cycle → ~6 Hz.
            // New pattern: 5 × 1 ms burst + ~28 ms sleep  → ~30 Hz.
            //
            // The 1 ms inter-request gap lets the TCP write buffer flush each
            // packet before the next is queued, preventing packet coalescing.
            while (true) {
                using ms = std::chrono::milliseconds;
                using clk = std::chrono::steady_clock;
                auto t0 = clk::now();

                _client->request(std::string(), 1000);
                std::this_thread::sleep_for(ms(1));
                _client->request(std::string(), 1001);
                std::this_thread::sleep_for(ms(1));
                _client->request(std::string(), 1002);
                std::this_thread::sleep_for(ms(1));
                _client->request(std::string(), 1003);
                std::this_thread::sleep_for(ms(1));
                _client->request(std::string(), 1004);

                // Sleep for whatever is left of the 33 ms window.
                auto used = std::chrono::duration_cast<ms>(clk::now() - t0).count();
                auto rem  = 33LL - used;
                if (rem > 0) std::this_thread::sleep_for(ms(rem));
            }
        });
        request_thread.detach();

        std::thread run_thread([this] {
            run();
        });
        run_thread.detach();
    }

protected:
    template <typename T>
    void publish(const T& msg) {
        const std::string topic = typeid(T).name();
        getBus().publish<T>(topic, msg);
    }

    template <typename T>
    void subscribe(T& sinkVar) {
        const std::string topic = typeid(T).name();
        getBus().subscribe<T>(topic, [&sinkVar](const T& incoming) {
            sinkVar = incoming;
        });
    }

    template <typename T>
    void subscribe(std::function<void(const T&)> callback) {
        const std::string topic = typeid(T).name();
        getBus().subscribe<T>(topic, callback);
    }

private:
    void process(const std::string data, uint16_t type)
    {
        if (type == 11000) {
            tmv::DepthFrame parsed;
            if (StringParsers::parseDepthFrame(data, parsed)) {
                std::lock_guard<std::mutex> lock(_mtxDepth);
                _depthFrame = std::move(parsed);
            }
        }
        else if (type == 11001) {
            tmv::ImageFrame parsed;
            if (StringParsers::parseImageFrame(data, parsed)) {
                std::lock_guard<std::mutex> lock(_mtxRgb);
                _rgbFrame = std::move(parsed);
            }
        }
        else if (type == 11002) {
            tmv::ImageFrame parsed;
            if (StringParsers::parseRGBDFrame(data, parsed)) {
                std::lock_guard<std::mutex> lock(_mtxRgbd);
                _rgbdFrame = std::move(parsed);
            }
        }
        else if (type == 11003) {
            tmv::MarkerDetections parsed;
            if (StringParsers::parseMarkerDetections(data, parsed)) {
                std::lock_guard<std::mutex> lock(_mtxMarkers);
                _markerDetections = std::move(parsed);
            }
        }
        else if (type == 11004) {
            tmv::SafetyStatus parsed;
            if (StringParsers::parseSafetyStatus(data, parsed)) {
                std::lock_guard<std::mutex> lock(_mtxSafety);
                _safetyStatus = std::move(parsed);
            }
        }
    }

    tmv::DepthFrame _depthFrame;
    tmv::ImageFrame _rgbFrame;
    tmv::ImageFrame _rgbdFrame;
    tmv::MarkerDetections _markerDetections;
    tmv::SafetyStatus _safetyStatus;

    std::mutex _mtxDepth;
    std::mutex _mtxRgb;
    std::mutex _mtxRgbd;
    std::mutex _mtxMarkers;
    std::mutex _mtxSafety;

    TCPClient *_client = nullptr;

    static MessageBus& getBus() {
        static MessageBus instance;
        return instance;
    }
};

#endif