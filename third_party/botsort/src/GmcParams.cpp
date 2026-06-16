// Motion-only build: see TrackerParams.cpp. Never-executed stub. Returns a
// default SparseOptFlow configuration for the requested method.
#include "GmcParams.h"

GMC_Params GMC_Params::load_config(GMC_Method method,
                                   const std::string& /*config_path*/) {
  GMC_Params params;
  params.method_ = method;
  params.method_params_ = SparseOptFlow_Params{};
  return params;
}
