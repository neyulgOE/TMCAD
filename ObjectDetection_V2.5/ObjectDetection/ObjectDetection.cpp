#include "ObjectDetection.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "BackendFactory.h"
#include "ObjectDetectionConfigValidator.h"
#include "ApplicationFactory.h"

ObjectDetection::ObjectDetection(const std::string& name)
    : Plugin(name)
{
    loadParams();

    setTopicCallBack<protocol::Message_CameraRGBD>(
        &ObjectDetection::message_CameraRGBD,
        this);

    createPublisher<protocol::Message_ObjectDetections>();

    initializeManager();
    initializeApplications();
}

ObjectDetection::~ObjectDetection()
{
    _running.store(false, std::memory_order_release);
}

void ObjectDetection::loadParams()
{
    const std::string group = "ObjectDetection";

    _config.backend = BackendFactory::compiledBackendName();

    getParamGroup(group, "enabled", _config.enabled, true);
    getParamGroup(group, "enable_perf_log", _config.enable_perf_log, true);
    getParamGroup(group, "perf_log_every_n", _config.perf_log_every_n, 300);
    getParamGroup(group,
                  "debug_publish_raw_objects",
                  _config.debug_publish_raw_objects,
                  false);
    getParamGroup(group, "loop_sleep_ms", _config.loop_sleep_ms, 20);

    getParamGroup(group, "backend", _config.backend, _config.backend);
    getParamGroup(group, "platform_profile", _config.platform_profile, std::string());
    getParamGroup(group,
                  "strict_config_validation",
                  _config.strict_config_validation,
                  _config.strict_config_validation);
    getParamGroup(group, "model_root", _config.model_root, _config.model_root);
    getParamGroup(group, "detectors", _config.detectors_text, std::string());
    getParamGroup(group, "source_name", _config.source_name, _config.source_name);
    getParamGroup(group, "image_encoding", _config.image_encoding, _config.image_encoding);

    getParamGroup(group,
                  "default_interval_ms",
                  _config.default_interval_ms,
                  _config.default_interval_ms);

    double default_conf = _config.default_conf_threshold;
    double default_nms = _config.default_nms_threshold;
    getParamGroup(group, "default_confidence_threshold", default_conf, default_conf);
    getParamGroup(group, "default_nms_threshold", default_nms, default_nms);
    _config.default_conf_threshold = static_cast<float>(default_conf);
    _config.default_nms_threshold = static_cast<float>(default_nms);

    getParamGroup(group,
                  "default_max_detections",
                  _config.default_max_detections,
                  _config.default_max_detections);

    _config.loop_sleep_ms = std::max(1, _config.loop_sleep_ms);
    _config.perf_log_every_n = std::max(1, _config.perf_log_every_n);
    _config.backend = lower(trim(_config.backend));
    _config.platform_profile = lower(trim(_config.platform_profile));
    _config.model_root = trim(_config.model_root);
    _config.source_name = trim(_config.source_name);
    _config.image_encoding = lower(trim(_config.image_encoding));
    _config.default_interval_ms = std::max(0, _config.default_interval_ms);
    _config.default_conf_threshold =
        std::max(0.0f, std::min(1.0f, _config.default_conf_threshold));
    _config.default_nms_threshold =
        std::max(0.0f, std::min(1.0f, _config.default_nms_threshold));
    _config.default_max_detections = std::max(1, _config.default_max_detections);

    _config.detector_ids = parseList(_config.detectors_text);
    _detector_configs.clear();
    _detector_configs.reserve(_config.detector_ids.size());
    for (const std::string& detector_id : _config.detector_ids)
        _detector_configs.push_back(loadDetectorParams(detector_id));
}

DetectorConfig ObjectDetection::loadDetectorParams(
    const std::string& detector_id)
{
    DetectorConfig config;
    config.detector_id = trim(detector_id);
    config.interval_ms = _config.default_interval_ms;
    config.conf_threshold = _config.default_conf_threshold;
    config.nms_threshold = _config.default_nms_threshold;
    config.max_detections = _config.default_max_detections;

    const std::string group = "ObjectDetection_" + detector_id;

    getParamGroup(group, "enabled", config.enabled, true);
    getParamGroup(group, "model_package", config.model_package, std::string());
    getParamGroup(group, "model_variant", config.model_variant, std::string("default"));
    getParamGroup(group, "interval_ms", config.interval_ms, config.interval_ms);

    double conf = config.conf_threshold;
    double nms = config.nms_threshold;
    getParamGroup(group, "confidence_threshold", conf, conf);
    getParamGroup(group, "nms_threshold", nms, nms);
    config.conf_threshold = static_cast<float>(conf);
    config.nms_threshold = static_cast<float>(nms);

    getParamGroup(group, "max_detections", config.max_detections, config.max_detections);
    getParamGroup(group, "enabled_classes", config.enabled_classes_text, std::string());
    getParamGroup(group, "application", config.application, config.application);

    config.model_package = trim(config.model_package);
    config.model_variant = lower(trim(config.model_variant));
    config.application = lower(trim(config.application));
    if (config.application.empty()) config.application = "none";

    config.interval_ms = std::max(0, config.interval_ms);
    config.conf_threshold = std::clamp(config.conf_threshold, 0.0f, 1.0f);
    config.nms_threshold = std::clamp(config.nms_threshold, 0.0f, 1.0f);
    config.max_detections = std::max(1, config.max_detections);

    config.enabled_classes = parseList(config.enabled_classes_text);
    for (std::string& label : config.enabled_classes)
        label = lower(trim(label));

    return config;
}

bool ObjectDetection::initializeManager()
{
    _initialized.store(false, std::memory_order_release);
    _model_ready.store(false, std::memory_order_release);

    if (!_config.enabled)
    {
        setStatus("disabled", "ObjectDetection disabled by configuration");
        _initialized.store(true, std::memory_order_release);
        return true;
    }

    std::vector<ObjectDetectionConfigIssue> config_issues;
    const bool config_valid = ObjectDetectionConfigValidator::validate(
        _config, _detector_configs, config_issues);

    for (const ObjectDetectionConfigIssue& issue : config_issues)
    {
        if (issue.fatal)
        {
            LogError("[ObjectDetection][ConfigError] detector="
                     << (issue.detector_id.empty() ? "global" : issue.detector_id)
                     << " code=" << issue.code
                     << " detail=" << issue.detail);
        }
        else
        {
            LogInfo("[ObjectDetection][ConfigWarning] detector="
                    << (issue.detector_id.empty() ? "global" : issue.detector_id)
                    << " code=" << issue.code
                    << " detail=" << issue.detail);
        }
    }

    if (!config_valid && _config.strict_config_validation)
    {
        setStatus("config_invalid",
                  "ObjectDetection configuration validation failed");
        _initialized.store(true, std::memory_order_release);
        return false;
    }

    const bool ready = _manager.initialize(_config, _detector_configs);
    _initialized.store(true, std::memory_order_release);
    _model_ready.store(ready, std::memory_order_release);

    if (ready)
    {
        setStatus("ready", "at least one detector is ready");
        LogInfo("[ObjectDetection][Init] ready"
                << " backend=" << _config.backend
                << " platform_profile=" << _config.platform_profile
                << " detectors=" << _manager.detectorCount()
                << " ready=" << _manager.readyCount()
                << " model_root=" << _config.model_root);
    }
    else
    {
        setStatus("no_ready_detector", "all configured detectors failed to initialize");
        LogError("[ObjectDetection][Init] no detector is ready. status="
                 << _manager.statusJson());
    }

    return ready;
}

bool ObjectDetection::initializeApplications()
{
    _applications.clear();
    _application_init_errors.clear();
    _requires_rgbd = false;

    bool all_requested_ready = true;

    for (const DetectorConfig& config : _detector_configs)
    {
        if (!config.enabled || config.application == "none")
            continue;

        std::string error;
        std::unique_ptr<IApplication> application =
            ApplicationFactory::create(config.application, error);

        if (!application)
        {
            _application_init_errors[config.detector_id] = error;
            all_requested_ready = false;
            LogError("[ObjectDetection][Application][Init] detector="
                     << config.detector_id << " " << error);
            continue;
        }

        if (!application->initialize(config, error))
        {
            _application_init_errors[config.detector_id] = error;
            all_requested_ready = false;
            LogError("[ObjectDetection][Application][Init] detector="
                     << config.detector_id
                     << " application=" << application->name()
                     << " error=" << error);
            continue;
        }

        _requires_rgbd = _requires_rgbd || application->requiresRgbd();

        LogInfo("[ObjectDetection][Application][Init] detector="
                << config.detector_id
                << " application=" << application->name()
                << " status=ready");

        _applications[config.detector_id] = std::move(application);
    }

    return all_requested_ready;
}

void ObjectDetection::message_CameraRGBD(google::protobuf::Message* /*data*/)
{
    _rgbd_received.store(true, std::memory_order_release);
    _has_new_rgbd.store(true, std::memory_order_release);
}

void ObjectDetection::run()
{
    _running.store(true, std::memory_order_release);

    while (_running.load(std::memory_order_acquire))
        SLEEP(_config.loop_sleep_ms);
}

bool ObjectDetection::detection()
{
    if (!_config.enabled) return true;

    std::unique_lock<std::mutex> lock(_detection_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return true;

    const uint64_t now_ns = rbs::core::Time::now().toNSec();
    if (!_initialized.load(std::memory_order_acquire) ||
        !_model_ready.load(std::memory_order_acquire))
    {
        if (shouldLogGuard(now_ns))
        {
            LogError("[ObjectDetection][Skip] no detector ready. status="
                     << statusJson());
        }
        return true;
    }

    // Process exactly once for each newly published Message_CameraRGBD.
    if (!_has_new_rgbd.exchange(false, std::memory_order_acq_rel))
        return true;

    protocol::Message_CameraRGBD camera_message;
    cv::Mat rgb;
    RgbdFrame rgbd_frame;
    uint64_t data_ns = 0;
    std::string rgbd_error;

    if (!readLatestRgbd(camera_message,
                        rgb,
                        rgbd_frame,
                        data_ns,
                        rgbd_error))
    {
        if (rgbd_error != "duplicate_rgbd_frame" &&
            shouldLogGuard(now_ns))
        {
            LogError("[ObjectDetection][RGBD] invalid: " << rgbd_error);
        }
        return true;
    }

    std::vector<DetectorOutput> outputs;
    _manager.process(rgb, now_ns, outputs);

    if (_config.debug_publish_raw_objects)
    {
        for (DetectorOutput& output : outputs)
            output.debug_raw_objects = output.result.objects;
    }

    _last_processed_data_ns = data_ns;
    _last_processed_color_frame_number = rgbd_frame.color_frame_number;

    applyApplications(&rgbd_frame, outputs);

    if (_config.debug_publish_raw_objects)
    {
        for (DetectorOutput& output : outputs)
        {
            if (!output.application_applied ||
                output.debug_raw_objects.empty())
            {
                continue;
            }

            output.result.objects.insert(
                output.result.objects.end(),
                output.debug_raw_objects.begin(),
                output.debug_raw_objects.end());
        }
    }

    for (const DetectorOutput& output : outputs)
        publishDetection(camera_message, data_ns, output);

    return true;
}

bool ObjectDetection::readLatestRgbd(
    protocol::Message_CameraRGBD& message,
    cv::Mat& rgb,
    RgbdFrame& frame,
    uint64_t& data_ns,
    std::string& error)
{
    getSubscriberData(message);

    const int width = static_cast<int>(message.width());
    const int height = static_cast<int>(message.height());
    if (width <= 0 || height <= 0)
    {
        error = "RGBD width/height is invalid";
        return false;
    }

    data_ns = static_cast<uint64_t>(message.header().data_nsec());
    const uint64_t color_frame_number = message.color_frame_number();

    if ((data_ns != 0 && data_ns == _last_processed_data_ns) ||
        (data_ns == 0 &&
         color_frame_number != 0 &&
         color_frame_number == _last_processed_color_frame_number))
    {
        error = "duplicate_rgbd_frame";
        return false;
    }

    uint32_t color_stride = message.color_stride_bytes();
    if (color_stride == 0)
        color_stride = message.stride_bytes();
    if (color_stride == 0)
        color_stride = static_cast<uint32_t>(width * 3);

    const uint32_t minimum_color_stride =
        static_cast<uint32_t>(width * 3);
    if (color_stride < minimum_color_stride)
    {
        error = "RGBD color stride is invalid";
        return false;
    }

    const size_t required_color_bytes =
        static_cast<size_t>(color_stride) * static_cast<size_t>(height - 1) +
        static_cast<size_t>(minimum_color_stride);
    if (message.bgr8().size() < required_color_bytes)
    {
        std::ostringstream ss;
        ss << "RGBD bgr8 buffer is too small: size="
           << message.bgr8().size()
           << " expected>=" << required_color_bytes;
        error = ss.str();
        return false;
    }

    cv::Mat bgr_view(height,
                     width,
                     CV_8UC3,
                     const_cast<char*>(message.bgr8().data()),
                     static_cast<size_t>(color_stride));

    if (_config.image_encoding == "bgr8")
    {
        cv::cvtColor(bgr_view, rgb, cv::COLOR_BGR2RGB);
    }
    else if (_config.image_encoding == "rgb8")
    {
        rgb = bgr_view.clone();
    }
    else
    {
        error = "unsupported image_encoding: " + _config.image_encoding;
        return false;
    }

    uint32_t depth_stride = message.depth_stride_bytes();
    const uint32_t minimum_depth_stride =
        static_cast<uint32_t>(width * sizeof(uint16_t));
    if (depth_stride == 0)
        depth_stride = minimum_depth_stride;

    if (depth_stride < minimum_depth_stride)
    {
        error = "RGBD depth stride is invalid";
        return false;
    }

    const size_t required_depth_bytes =
        static_cast<size_t>(depth_stride) * static_cast<size_t>(height - 1) +
        static_cast<size_t>(minimum_depth_stride);
    if (message.depth_z16().size() < required_depth_bytes)
    {
        std::ostringstream ss;
        ss << "RGBD depth_z16 buffer is too small: size="
           << message.depth_z16().size()
           << " expected>=" << required_depth_bytes;
        error = ss.str();
        return false;
    }

    cv::Mat depth_view(height,
                       width,
                       CV_16UC1,
                       const_cast<char*>(message.depth_z16().data()),
                       static_cast<size_t>(depth_stride));

    frame.depth_z16 = depth_view.clone();
    frame.rgb = rgb.clone(); // V2.4: forward RGB image to applications
    frame.width = message.width();
    frame.height = message.height();
    frame.color_stride_bytes = color_stride;
    frame.depth_stride_bytes = depth_stride;
    frame.depth_scale = static_cast<double>(message.depth_scale());

    if (frame.depth_scale <= 0.0)
    {
        error = "RGBD depth_scale is invalid";
        return false;
    }

    if (!message.has_intr())
    {
        error = "RGBD intrinsics are missing";
        return false;
    }

    const auto& intr = message.intr();
    frame.intrinsics.fx = intr.fx();
    frame.intrinsics.fy = intr.fy();
    frame.intrinsics.ppx = intr.ppx();
    frame.intrinsics.ppy = intr.ppy();

    if (intr.width() != 0 && intr.width() != message.width())
    {
        error = "RGBD intrinsics width does not match image width";
        return false;
    }
    if (intr.height() != 0 && intr.height() != message.height())
    {
        error = "RGBD intrinsics height does not match image height";
        return false;
    }

    frame.rgb_timestamp_ms = message.color_rs_timestamp_ms();
    if (frame.rgb_timestamp_ms <= 0.0)
        frame.rgb_timestamp_ms = message.rs_timestamp_ms();

    frame.depth_timestamp_ms = message.depth_rs_timestamp_ms();
    if (frame.depth_timestamp_ms <= 0.0)
        frame.depth_timestamp_ms = message.rs_timestamp_ms();

    frame.color_frame_number = color_frame_number;
    frame.depth_frame_number = message.depth_frame_number();
    frame.alignment = static_cast<int>(message.alignment());
    frame.color_timestamp_domain = message.color_timestamp_domain();
    frame.depth_timestamp_domain = message.depth_timestamp_domain();
    frame.data_ns = data_ns;

    if (!frame.valid())
    {
        error = "RGBD frame is invalid";
        return false;
    }

    if (!_logged_rgbd_metadata)
    {
        _logged_rgbd_metadata = true;
        LogInfo("[ObjectDetection][RGBD] source=Message_CameraRGBD"
                << " size=" << frame.width << "x" << frame.height
                << " color_stride=" << frame.color_stride_bytes
                << " depth_stride=" << frame.depth_stride_bytes
                << " depth_scale=" << frame.depth_scale
                << " alignment=" << frame.alignment
                << " color_frame=" << frame.color_frame_number
                << " depth_frame=" << frame.depth_frame_number
                << " color_ts_ms=" << frame.rgb_timestamp_ms
                << " depth_ts_ms=" << frame.depth_timestamp_ms
                << " intr=[" << frame.intrinsics.fx
                << "," << frame.intrinsics.fy
                << "," << frame.intrinsics.ppx
                << "," << frame.intrinsics.ppy << "]");
    }

    error.clear();
    return true;
}

void ObjectDetection::applyApplications(
    const RgbdFrame* frame,
    std::vector<DetectorOutput>& outputs)
{
    for (DetectorOutput& output : outputs)
    {
        const auto app_it = _applications.find(output.detector_id);
        if (app_it == _applications.end())
        {
            const auto error_it = _application_init_errors.find(output.detector_id);
            if (error_it != _application_init_errors.end())
            {
                output.application_requested = true;
                output.application_applied = false;
                output.application_status = "application_not_ready";
                output.result.objects.clear();
            }
            continue;
        }

        output.application_requested = true;
        output.application = app_it->second->name();

        if (app_it->second->requiresRgbd() && !frame)
        {
            output.application_applied = false;
            output.application_status = "rgbd_unavailable";
            continue;
        }

        std::string error;
        if (!app_it->second->process(*frame, output, error))
        {
            output.application_applied = false;
            if (output.application_status.empty())
                output.application_status = "application_failed";

            uint64_t& count = _detection_counters[output.detector_id + "_application_error"];
            ++count;
            if (count == 1 || count % 30ULL == 0ULL)
            {
                LogError("[ObjectDetection][Application][Run] detector="
                         << output.detector_id
                         << " application=" << output.application
                         << " status=" << output.application_status
                         << " error=" << error);
            }
        }
    }
}

void ObjectDetection::publishDetection(
    const protocol::Message_CameraRGBD& camera_message,
    uint64_t data_ns,
    const DetectorOutput& output)
{
    protocol::Message_ObjectDetections message;
    const uint64_t publish_ns = rbs::core::Time::now().toNSec();

    message.mutable_header()->set_pub_nsec(publish_ns);
    message.mutable_header()->set_data_nsec(data_ns != 0 ? data_ns : publish_ns);

    message.set_width(static_cast<uint32_t>(output.result.image_width));
    message.set_height(static_cast<uint32_t>(output.result.image_height));
    message.set_source_name(_config.source_name);
    message.set_model_name(output.model_name);
    message.set_model_id(output.model_id);
    double rgb_timestamp_ms = camera_message.color_rs_timestamp_ms();
    if (rgb_timestamp_ms <= 0.0)
        rgb_timestamp_ms = camera_message.rs_timestamp_ms();
    message.set_rgb_timestamp_ms(rgb_timestamp_ms);
    message.set_depth_timestamp_ms(output.depth_timestamp_ms);
    message.set_depth_ms(static_cast<float>(output.application_ms));

    message.set_preprocess_ms(static_cast<float>(output.result.preprocess_ms));
    message.set_inference_ms(static_cast<float>(output.result.inference_ms));
    message.set_postprocess_ms(static_cast<float>(output.result.postprocess_ms));
    message.set_total_ms(static_cast<float>(output.result.total_ms));

    for (const DetectionObject& object : output.result.objects)
    {
        auto* item = message.add_objects();
        item->set_class_id(object.class_id);
        item->set_label(object.label);
        item->set_confidence(object.confidence);
        item->set_x(object.x);
        item->set_y(object.y);
        item->set_width(object.width);
        item->set_height(object.height);
        item->set_center_x(object.center_x);
        item->set_center_y(object.center_y);
        item->set_track_id(-1);

        item->set_has_3d(object.has_3d);

        if (object.has_3d)
        {
            item->set_position_x_m(object.x_3d);
            item->set_position_y_m(object.y_3d);
            item->set_position_z_m(object.z_3d);
            item->set_depth_m(object.depth_m > 0.0f
                                  ? object.depth_m
                                  : object.z_3d);
            item->set_depth_valid_ratio(std::clamp(
                object.depth_valid_ratio, 0.0f, 1.0f));
            item->set_depth_std_m(std::max(0.0f, object.depth_std_m));
        }
        else
        {
            item->set_position_x_m(0.0f);
            item->set_position_y_m(0.0f);
            item->set_position_z_m(0.0f);
            item->set_depth_m(0.0f);
            item->set_depth_valid_ratio(0.0f);
            item->set_depth_std_m(0.0f);
        }

        item->set_has_pose_6d(object.has_pose_6d);
        if (object.has_pose_6d)
        {
            item->set_roll_deg(object.roll_deg);
            item->set_pitch_deg(object.pitch_deg);
            item->set_yaw_deg(object.yaw_deg);
            item->set_plane_normal_x(object.plane_normal_x);
            item->set_plane_normal_y(object.plane_normal_y);
            item->set_plane_normal_z(object.plane_normal_z);
            item->set_plane_inlier_ratio(std::clamp(
                object.plane_inlier_ratio, 0.0f, 1.0f));
            item->set_plane_rmse_m(std::max(0.0f, object.plane_rmse_m));
            item->set_pair_distance_m(std::max(0.0f, object.pair_distance_m));
            item->set_plane_source(object.plane_source);
            item->set_spatial_status(object.spatial_status);
            item->set_pose_frame_id(object.pose_frame_id);
        }
        else
        {
            item->set_roll_deg(0.0f);
            item->set_pitch_deg(0.0f);
            item->set_yaw_deg(0.0f);
            item->set_plane_normal_x(0.0f);
            item->set_plane_normal_y(0.0f);
            item->set_plane_normal_z(0.0f);
            item->set_plane_inlier_ratio(0.0f);
            item->set_plane_rmse_m(0.0f);
            item->set_pair_distance_m(0.0f);
            item->set_plane_source(object.plane_source);
            item->set_spatial_status(object.spatial_status);
            item->set_pose_frame_id(object.pose_frame_id);
        }
    }

    publishTopic(message);

    uint64_t& count = _detection_counters[output.detector_id];
    ++count;
    const uint64_t perf_every = static_cast<uint64_t>(_config.perf_log_every_n);
    if (_config.enable_perf_log &&
        (count == 1ULL || count % perf_every == 0ULL))
    {
        LogInfo("[ObjectDetection][Perf] detector=" << output.detector_id
                << " object_type=" << output.object_type
                << " task_type=" << output.task_type
                << " platform_profile=" << output.platform_profile
                << " model=" << output.model_id
                << " backend=" << output.backend
                << " variant=" << output.model_variant
                << " count=" << count
                << " objects=" << output.result.objects.size()
                << " xyz=" << output.xyz_object_count
                << " pose6d=" << output.pose_object_count
                << " application="
                << (output.application.empty() ? "none" : output.application)
                << " application_status="
                << (output.application_status.empty() ? "2d_only" : output.application_status)
                << " pre_ms=" << output.result.preprocess_ms
                << " infer_ms=" << output.result.inference_ms
                << " post_ms=" << output.result.postprocess_ms
                << " application_ms=" << output.application_ms
                << " total_ms=" << output.result.total_ms);
    }

}

bool ObjectDetection::isModelReady() const
{
    return _model_ready.load(std::memory_order_acquire);
}

bool ObjectDetection::isInitialized() const
{
    return _initialized.load(std::memory_order_acquire);
}

std::string ObjectDetection::statusJson() const
{
    std::lock_guard<std::mutex> lock(_status_mutex);

    std::ostringstream ss;
    ss << '{';
    ss << "\"ok\":true";
    ss << ",\"task\":\"object_detection\"";
    ss << ",\"enabled\":" << (_config.enabled ? "true" : "false");
    ss << ",\"initialized\":"
       << (_initialized.load(std::memory_order_acquire) ? "true" : "false");
    ss << ",\"ready\":"
       << (_model_ready.load(std::memory_order_acquire) ? "true" : "false");
    ss << ",\"backend\":" << jsonEscape(_config.backend);
    ss << ",\"compiled_backend\":"
       << jsonEscape(BackendFactory::compiledBackendName());
    ss << ",\"model_root\":" << jsonEscape(_config.model_root);
    ss << ",\"source_name\":" << jsonEscape(_config.source_name);
    ss << ",\"image_encoding\":" << jsonEscape(_config.image_encoding);
    ss << ",\"debug_publish_raw_objects\":"
       << (_config.debug_publish_raw_objects ? "true" : "false");
    ss << ",\"requires_rgbd\":" << (_requires_rgbd ? "true" : "false");
    ss << ",\"rgbd_received\":"
       << (_rgbd_received.load(std::memory_order_acquire) ? "true" : "false");
    ss << ",\"input_contract\":\"Message_CameraRGBD.bgr8+depth_z16+intr\"";
    ss << ",\"applications\":[";

    bool first = true;
    for (const auto& item : _applications)
    {
        if (!first) ss << ',';
        first = false;
        ss << '{';
        ss << "\"detector_id\":" << jsonEscape(item.first);
        ss << ",\"ready\":true";
        ss << ",\"name\":" << jsonEscape(item.second->name());
        ss << '}';
    }
    for (const auto& item : _application_init_errors)
    {
        if (!first) ss << ',';
        first = false;
        ss << '{';
        ss << "\"detector_id\":" << jsonEscape(item.first);
        ss << ",\"ready\":false";
        ss << ",\"error\":" << jsonEscape(item.second);
        ss << '}';
    }
    ss << ']';
    ss << ",\"status_code\":" << jsonEscape(_status_code);
    ss << ",\"status_detail\":" << jsonEscape(_status_detail);
    ss << ",\"manager\":" << _manager.statusJson();
    ss << '}';
    return ss.str();
}

std::vector<std::string> ObjectDetection::parseList(const std::string& text) const
{
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        token = trim(token);
        if (!token.empty() &&
            std::find(values.begin(), values.end(), token) == values.end())
        {
            values.push_back(token);
        }
    }
    return values;
}

std::string ObjectDetection::trim(const std::string& value)
{
    size_t first = 0;
    while (first < value.size() &&
           std::isspace(static_cast<unsigned char>(value[first])))
    {
        ++first;
    }

    size_t last = value.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(value[last - 1])))
    {
        --last;
    }
    return value.substr(first, last - first);
}

std::string ObjectDetection::lower(const std::string& value)
{
    std::string output = value;
    std::transform(output.begin(), output.end(), output.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return output;
}

std::string ObjectDetection::jsonEscape(const std::string& value)
{
    std::ostringstream ss;
    ss << '"';
    for (char c : value)
    {
        switch (c)
        {
        case '\\': ss << "\\\\"; break;
        case '"': ss << "\\\""; break;
        case '\n': ss << "\\n"; break;
        case '\r': ss << "\\r"; break;
        case '\t': ss << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                ss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                   << static_cast<int>(static_cast<unsigned char>(c));
            }
            else
            {
                ss << c;
            }
        }
    }
    ss << '"';
    return ss.str();
}

void ObjectDetection::setStatus(const std::string& code,
                                const std::string& detail)
{
    std::lock_guard<std::mutex> lock(_status_mutex);
    _status_code = code;
    _status_detail = detail;
}

bool ObjectDetection::shouldLogGuard(uint64_t now_ns)
{
    const uint64_t interval_ns = 5ULL * 1000ULL * 1000ULL * 1000ULL;
    if (_last_guard_log_ns == 0 || now_ns < _last_guard_log_ns ||
        now_ns - _last_guard_log_ns >= interval_ns)
    {
        _last_guard_log_ns = now_ns;
        return true;
    }
    return false;
}
