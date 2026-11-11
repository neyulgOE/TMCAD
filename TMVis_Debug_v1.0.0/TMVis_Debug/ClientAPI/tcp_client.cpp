#include "tcp_client.h"
#include <iostream>
#include <thread>
#include <chrono>

class NetProtocolHeader{
public:
    NetProtocolHeader(){
        m_header = 0x5A;
        m_version = 0x01;
        m_number = 0;
        m_length = 0;
        m_type = 0;
        memset(m_reserved,0, 6);
    }

    uint8_t m_header;
    uint8_t m_version;
    uint16_t m_number;
    uint32_t m_length;
    uint16_t m_type;
    uint8_t m_reserved[6];
};

TCPClient::TCPClient() : _socket(_io_service){

}

TCPClient::~TCPClient(){
    _socket.close();
    _io_service.stop();
    if(boost::this_thread::get_id() != _io_service_thread.get_id())
        _io_service_thread.join();
}

void TCPClient::start(){
    try {
        boost::asio::ip::tcp::resolver resolver(_io_service);
        boost::asio::ip::tcp::resolver::query query("192.168.1.111", "19210");

        boost::system::error_code error = boost::asio::error::host_not_found;
        int retry = 0;
        while (error && retry < 5) {
            boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve(query, error);
            boost::asio::ip::tcp::resolver::iterator end;
            while (!error && endpoint_iterator != end) {
                _socket.close();
                _socket.connect(*endpoint_iterator++, error);
                if (!error) break;
            }
            if (error) {
                std::cerr << "[TCPClient] Connect to 192.168.1.111:19210 error: " << error.message()
                          << ", retrying (" << (retry + 1) << "/5)..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(800));
                ++retry;
            }
        }
        if (error) {
            std::cerr << "[TCPClient] Connection failed permanently: " << error.message() << std::endl;
            return;
        }

        _socket.async_read_some(boost::asio::buffer(_readQueueBuffer, 65536), boost::bind(&TCPClient::handleResponse, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
        _io_service_thread = boost::thread(boost::bind(&boost::asio::io_service::run, &_io_service));
    } catch (const std::exception& e) {
        std::cerr << "[TCPClient] Exception in start(): " << e.what() << std::endl;
    }
}

void TCPClient::write(const char* data, const size_t size){
    boost::lock_guard<boost::mutex> lock(_writeMutex);
    _writeBuffer.insert(_writeBuffer.end(), data, data + size);
    doWrite();
}

void TCPClient::doWrite(){
    _socket.async_write_some(boost::asio::null_buffers(), boost::bind(&TCPClient::writeEnd, this, boost::asio::placeholders::error));
}

void TCPClient::writeEnd(const boost::system::error_code &ec){
    if(_writeBuffer.empty()){
        return;
    }
    if(!ec){
        std::size_t len = _socket.write_some(boost::asio::buffer(_writeBuffer, _writeBuffer.size()));
        //std::cout << "len: " << len << std::endl;
        _writeBuffer.erase(_writeBuffer.begin(), _writeBuffer.begin() + len);
    }
    if(_writeBuffer.size()){
        doWrite();
    }

}

void TCPClient::request(const std::string data, uint16_t type){
    if(_socket.is_open()){
        char* dataRequest = nullptr;
        int size = 0;
        if(data.empty()){
            size = sizeof(NetProtocolHeader);
            dataRequest = new char[size];
            NetProtocolHeader* header = new NetProtocolHeader;
            header->m_header = 0x5A;
            header->m_version = 0x01;
            header->m_type = type;
            header->m_number = 0x00;
            header->m_length = 0;
            memset(header->m_reserved, 0, 6);
            memcpy(dataRequest, header, sizeof(NetProtocolHeader));
            delete header;
        }else{
            size = sizeof(NetProtocolHeader) + data.length();
            dataRequest = new char[size];
            NetProtocolHeader* header = new NetProtocolHeader;
            header->m_header = 0x5A;
            header->m_version = 0x01;
            header->m_type = type;
            header->m_number = 0x00;
            header->m_length = data.length();
            memset(header->m_reserved, 0, 6);
            memcpy(dataRequest, header, sizeof(NetProtocolHeader));
            delete header;
            memcpy(dataRequest + sizeof(NetProtocolHeader), data.data(), data.length());
        }
        //std::cout << "size: " << size << std::endl;
        write(dataRequest, size);
        delete[] dataRequest;
    }
}

void TCPClient::handleResponse(const boost::system::error_code &err, size_t bytes_transferred){
    if(!err && bytes_transferred > 0){
        _readBuffer.insert(_readBuffer.end(), _readQueueBuffer, _readQueueBuffer + bytes_transferred);
        while(_readBuffer.size() > 0){
            char a0 = _readBuffer.at(0);
            if(uint8_t(a0) == 0x5A){
                if(_readBuffer.size() >= 16){
                    NetProtocolHeader* header = new NetProtocolHeader;
                    memcpy(header, _readBuffer.data(), 16);
                    uint32_t length = header->m_length;
                    uint16_t type = header->m_type;
                    delete header;
                    if(length > _readBuffer.size() - 16){
                        break;
                    }else{
                        std::string response(_readBuffer.data() + 16, _readBuffer.data() + 16 + length);
                        _callback(response, type);
                        _readBuffer.erase(_readBuffer.begin(), _readBuffer.begin() + 16 + length);
                    }
                }else{
                    break;
                }
            }else{
                _readBuffer.erase(_readBuffer.begin(), _readBuffer.begin() + 1);
            }
            
        }
        _socket.async_read_some(boost::asio::buffer(_readQueueBuffer, 65536), boost::bind(&TCPClient::handleResponse, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
    }else{
        _socket.close();
    }
}

