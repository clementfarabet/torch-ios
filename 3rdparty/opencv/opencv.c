
#include <TH.h>
#include <luaT.h>

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define libopencv_(NAME) TH_CONCAT_3(libopencv_, Real, NAME)

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;


#include "generic/opencv.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libopencv(lua_State *L)
{
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  libopencv_FloatMain_init(L);
  libopencv_DoubleMain_init(L);

  luaL_register(L, "libopencv.double", libopencv_DoubleMain__);
  luaL_register(L, "libopencv.float", libopencv_FloatMain__);

  return 1;
}
