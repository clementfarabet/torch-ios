#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialUpSampling.c"
#else

static int nn_(SpatialUpSampling_updateOutput)(lua_State *L)
{
  // get all params
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  // dims
  int iwidth = input->size[2];
  int iheight = input->size[1];
  int ochannels = input->size[0];
  int owidth = iwidth * dW;
  int oheight = iheight * dH;

  // get strides
  long *is = input->stride;
  long *os = output->stride;

  // get raw pointers
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);

  // resample each plane
  int k;
  for (k=0; k<ochannels; k++) {
    // get planes
    real *input_p = input_data + k*is[0];
    real *output_p = output_data + k*os[0];

    // for each plane, resample
    int x,y;
    for (y=0; y<oheight; y++) {
      for (x=0; x<owidth; x++) {
        // input positions (floored)
        int ix = x/dW;
        int iy = y/dH;

        // set output
        output_p[y*os[1] + x*os[2]] = input_p[iy*is[1] + ix*is[2]];
      }
    }
  }
  return 1;
}

static int nn_(SpatialUpSampling_updateGradInput)(lua_State *L)
{
  // get all params
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  // dims
  int iwidth = input->size[2];
  int iheight = input->size[1];
  int ichannels = input->size[0];
  int owidth = gradOutput->size[2];
  int oheight = gradOutput->size[1];
  int ochannels = gradOutput->size[0];

  // resize gradInput
  THTensor_(zero)(gradInput);

  // get strides
  long *gis = gradInput->stride;
  long *gos = gradOutput->stride;


  // get raw pointers
  real *gradInput_data = THTensor_(data)(gradInput);
  real *gradOutput_data = THTensor_(data)(gradOutput);

  // compute gradients for each plane
  int k;
  for (k=0; k<ochannels; k++) {
    // get planes
    real *gradInput_p = gradInput_data + k*gis[0];
    real *gradOutput_p = gradOutput_data + k*gos[0];

    // for each plane, resample
    int x,y;
    for (y=0; y<oheight; y++) {
      for (x=0; x<owidth; x++) {
        // input positions (floored)
        int ix = x/dW;
        int iy = y/dH;

        // accumulate gradient
        gradInput_p[iy*gis[1] + ix*gis[2]] += gradOutput_p[y*gos[1] + x*gos[2]];
      }
    }
  }
  return 1;
}

static const struct luaL_Reg nn_(SpatialUpSampling__) [] = {
  {"SpatialUpSampling_updateOutput", nn_(SpatialUpSampling_updateOutput)},
  {"SpatialUpSampling_updateGradInput", nn_(SpatialUpSampling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialUpSampling_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialUpSampling__), "nn");
  lua_pop(L,1);
}

#endif
