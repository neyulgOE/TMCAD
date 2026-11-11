#ifndef TCP_CLIENT_H
#define TCP_CLIENT_H

#include <boost/asio.hpp>
#include <boost/thread.hpp>

class TCPClient {
public:
	TCPClient();

	~TCPClient();

    void start();

    void write(const char* data, const size_t size);

    void doWrite();

    void writeEnd(const boost::system::error_code &ec);

	void request(const std::string data, uint16_t type);

    void setProcessCallback(const boost::function<void(const std::string, uint16_t)> &cb){
        _callback = cb;
    }

private:
	void handleResponse(const boost::system::error_code &err, size_t bytes_transferred);

private:
    boost::function<void(const std::string, uint16_t)> _callback;
    boost::asio::io_service _io_service;
    boost::asio::ip::tcp::socket _socket;
    boost::thread _io_service_thread;
    char _readQueueBuffer[65536];
    std::vector<char> _readBuffer;
    std::vector<char> _writeBuffer;
    boost::mutex _writeMutex;
};

#endif