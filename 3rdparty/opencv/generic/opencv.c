#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/opencv.c"
#else

//======================================================================
// File: opencv
//
// Description: A wrapper for a couple of OpenCV functions.
//
// Created: February 12, 2010, 1:22AM
//
// Install on ubuntu :
//  http://www.samontab.com/web/2010/04/installing-opencv-2-1-in-ubuntu/
//
// Author: Clement Farabet // clement.farabet@gmail.com
//         Marco Scoffier // github@metm.org (more functions)
//======================================================================

#include <luaT.h>
#include <TH.h>

#include <dirent.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <signal.h>
#include <pthread.h>

#include <cv.h>
#include <highgui.h>

#define CV_NO_BACKWARD_COMPATIBILITY

static void libopencv_(Main_opencvMat2torch)(CvMat *source, THTensor *dest) {

  int mat_step;
  CvSize mat_size;
  THTensor *tensor;
  // type dependent variables
  float * data_32F;
  float * data_32Fp;
  double * data_64F;
  double * data_64Fp;
  uchar * data_8U;
  uchar * data_8Up;
  unsigned int * data_16U;
  unsigned int * data_16Up;
  switch (CV_MAT_DEPTH(source->type))
    {
    case CV_32F:
      cvGetRawData(source, (uchar**)&data_32F, &mat_step, &mat_size);
      // Resize target
      THTensor_(resize2d)(dest, source->rows, source->cols);
      tensor = THTensor_(newContiguous)(dest);
      data_32Fp = data_32F;
      // copy
      TH_TENSOR_APPLY(real, tensor,
                      *tensor_data = ((real)(*data_32Fp));
                      // step through channels of ipl
                      data_32Fp++;
                      );
      THTensor_(free)(tensor);
      break;
    case CV_64F:
      cvGetRawData(source, (uchar**)&data_64F, &mat_step, &mat_size);
      // Resize target
      THTensor_(resize2d)(dest, source->rows, source->cols);
      tensor = THTensor_(newContiguous)(dest);

      data_64Fp = data_64F;
      // copy
      TH_TENSOR_APPLY(real, tensor,
                      *tensor_data = ((real)(*data_64Fp));
                      // step through channels of ipl
                      data_64Fp++;
                      );
      THTensor_(free)(tensor);
      break;
    case CV_8U:
      cvGetRawData(source, (uchar**)&data_8U, &mat_step, &mat_size);
      // Resize target
      THTensor_(resize2d)(dest, source->rows, source->cols);
      tensor = THTensor_(newContiguous)(dest);

      data_8Up = data_8U;
      // copy
      TH_TENSOR_APPLY(real, tensor,
                      *tensor_data = ((real)(*data_8Up));
                      // step through channels of ipl
                      data_8Up++;
                      );
      THTensor_(free)(tensor);
      break;
    case CV_16U:
      cvGetRawData(source, (uchar**)&data_16U, &mat_step, &mat_size);
      // Resize target
      THTensor_(resize2d)(dest, source->rows, source->cols);
      tensor = THTensor_(newContiguous)(dest);

      data_16Up = data_16U;
      // copy
      TH_TENSOR_APPLY(real, tensor,
                      *tensor_data = ((real)(*data_16Up));
                      // step through channels of ipl
                      data_16Up++;
                      );
      THTensor_(free)(tensor);
      break;
    default:
      THError("invalid CvMat type");
      break;
    }
}

static void libopencv_(Main_opencv8U2torch)(IplImage *source, THTensor *dest) {
  // Pointers
  uchar * source_data;

  // Get pointers / info
  int source_step;
  CvSize source_size;
  cvGetRawData(source, (uchar**)&source_data, &source_step, &source_size);

  // Resize target
  THTensor_(resize3d)(dest, source->nChannels, source->height, source->width);
  THTensor *tensor = THTensor_(newContiguous)(dest);

  // Torch stores channels first, opencv last so we select the channel
  // in torch tensor and step through the opencv iplimage.
  int j = 0;
  int k = source->nChannels-1;
  uchar * sourcep = source_data;
  for (j=0;j<source->nChannels;j++){
    sourcep = source_data+k-j; // start at correct channel opencv is BGR
    THTensor *tslice = THTensor_(newSelect)(tensor,0,j);
    // copy
    TH_TENSOR_APPLY(real, tslice,
		    *tslice_data = ((real)(*sourcep))/255.0;
		    // step through channels of ipl
		    sourcep = sourcep + source->nChannels;
		    );
    THTensor_(free)(tslice);
  }

  // cleanup
  THTensor_(free)(tensor);
}

static void libopencv_(Main_opencv32F2torch)(IplImage *source, THTensor *dest) {
  // Pointers
  float * source_data;

  // Get pointers / info
  int source_step;
  CvSize source_size;
  cvGetRawData(source, (uchar**)&source_data, &source_step, &source_size);

  // Resize target
  THTensor_(resize3d)(dest, source->nChannels, source->height, source->width);
  THTensor *tensor = THTensor_(newContiguous)(dest);

  // Torch stores channels first, opencv last so we select the channel
  // in torch tensor and step through the opencv iplimage.
  int j = 0;
  int k = source->nChannels-1;
  float * sourcep = source_data;
  for (j=0;j<source->nChannels;j++){
    sourcep = source_data+k-j; // start at correct channel opencv is BGR
    THTensor *tslice = THTensor_(newSelect)(tensor,0,j);
    // copy
    TH_TENSOR_APPLY(real, tslice,
		    *tslice_data = (real)(*sourcep);
		    // step through ipl
		    sourcep = sourcep + source->nChannels;
		    );
    THTensor_(free)(tslice);
  }

  // cleanup
  THTensor_(free)(tensor);
}

static IplImage * libopencv_(Main_torchimg2opencv_8U)(THTensor *source) {
  // Pointers
  uchar * dest_data;

  // Get size and channels
  int channels = source->size[0];
  int dest_step;
  CvSize dest_size = cvSize(source->size[2], source->size[1]);

  // Create ipl image
  IplImage * dest = cvCreateImage(dest_size, IPL_DEPTH_8U, channels);

  // get pointer to raw data
  cvGetRawData(dest, (uchar**)&dest_data, &dest_step, &dest_size);

  // copy
  THTensor *tensor = THTensor_(newContiguous)(source);

  // Torch stores channels first, opencv last so we select the channel
  // in torch tensor and step through the opencv iplimage.
  int j = 0;
  int k = channels-1;
  uchar * destp = dest_data;
  for (j=0;j<dest->nChannels;j++){
    destp = dest_data+k-j; // start at correct channel opencv is BGR
    THTensor *tslice = THTensor_(newSelect)(tensor,0,j);
    // copy
    TH_TENSOR_APPLY(real, tslice,
		    *destp = (uchar)(*tslice_data * 255.0);
		    // step through ipl
		    destp = destp + dest->nChannels;
		    );
    THTensor_(free)(tslice);
  }

  // free
  THTensor_(free)(tensor);

  // return freshly created IPL image
  return dest;
}

static IplImage * libopencv_(Main_torchimg2opencv_32F)(THTensor *source) {
  // Pointers
  float * dest_data;

  // Get size and channels
  int channels = source->size[0];
  int dest_step;
  CvSize dest_size = cvSize(source->size[2], source->size[1]);

  // Create ipl image
  IplImage * dest = cvCreateImage(dest_size, IPL_DEPTH_32F, channels);

  // get pointer to raw data
  cvGetRawData(dest, (uchar**)&dest_data, &dest_step, &dest_size);
  // copy
  THTensor *tensor = THTensor_(newContiguous)(source);
  // Torch stores channels first, opencv last so we select the channel
  // in torch tensor and step through the opencv iplimage.
  int j = 0;
  int k = channels-1;
  float * destp = dest_data;
  for (j=0;j<dest->nChannels;j++){
    destp = dest_data+k-j; // start at correct channel opencv is BGR
    THTensor *tslice = THTensor_(newSelect)(tensor,0,j);
    // copy
    TH_TENSOR_APPLY(real, tslice,
		    *destp = (float)(*tslice_data);
		    destp = destp + dest->nChannels; // step through ipl
		    );
    THTensor_(free)(tslice);
  }
  THTensor_(free)(tensor);
  // return freshly created IPL image
  return dest;
}

static void libopencv_(Main_opencvPoints2torch)(CvPoint2D32f * points, int npoints, THTensor *tensor) {

  // Resize target
  THTensor_(resize2d)(tensor, npoints, 2);
  THTensor *tensorc = THTensor_(newContiguous)(tensor);
  real *tensor_data = THTensor_(data)(tensorc);

  // copy
  int p;
  for (p=0; p<npoints; p++){
    *tensor_data++ = (real)points[p].x;
    *tensor_data++ = (real)points[p].y;
  }

  // cleanup
  THTensor_(free)(tensorc);
}

static CvPoint2D32f * libopencv_(Main_torch2opencvPoints)(THTensor *src) {

  int count = src->size[0];
  // create output
  CvPoint2D32f * points_cv = NULL;
  points_cv = (CvPoint2D32f*)cvAlloc(count*sizeof(points_cv[0]));
  real * src_pt = THTensor_(data)(src);
  // copy
  int p;
  for (p=0; p<count; p++){
    points_cv[p].x = (float)*src_pt++ ;
    points_cv[p].y = (float)*src_pt++ ;
  }

  // return freshly created CvPoint2D32f
  return points_cv;
}


//============================================================
static int libopencv_(Main_cvCornerHarris) (lua_State *L) {
  // Get Tensor's Info
  THTensor * image  = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * harris = luaT_checkudata(L, 2, torch_(Tensor_id));

  if (image->size[0] > 1){
    printf("WARNING: CorverHarris only accepts single channel images\n");
  } else {
    CvSize dest_size = cvSize(image->size[2], image->size[1]);
    IplImage * image_ipl = libopencv_(Main_torchimg2opencv_8U)(image);
    // Create ipl image
    IplImage * harris_ipl = cvCreateImage(dest_size, IPL_DEPTH_32F, 1);
    int blockSize = 5;
    int aperture_size = 3;
    double k = 0.04;

    // User values:
    if (lua_isnumber(L, 3)) {
      blockSize = lua_tonumber(L, 3);
    }
    if (lua_isnumber(L, 4)) {
      aperture_size = lua_tonumber(L, 4);
    }
    if (lua_isnumber(L, 5)) {
      k = lua_tonumber(L, 5);
    }

    cvCornerHarris(image_ipl, harris_ipl, blockSize, aperture_size, k);

    // return results
    libopencv_(Main_opencv32F2torch)(harris_ipl, harris);

    // Deallocate IPL images
    cvReleaseImage(&harris_ipl);
    cvReleaseImage(&image_ipl);
  }

  return 0;
}

//============================================================
// OpticalFlow
// Works on torch.Tensors (double). All the conversions are
// done in C.
//
static int libopencv_(Main_cvCalcOpticalFlow)(lua_State *L) {
  // Get Tensor's Info
  THTensor * curr = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * prev = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor * velx = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor * vely = luaT_checkudata(L, 4, torch_(Tensor_id));

  // Generate IPL images
  IplImage * curr_ipl = libopencv_(Main_torchimg2opencv_8U)(curr);
  IplImage * prev_ipl = libopencv_(Main_torchimg2opencv_8U)(prev);
  IplImage * velx_ipl;
  IplImage * vely_ipl;

  // Default values
  int method = 1;
  int lagrangian = 1;
  int iterations = 5;
  CvSize blockSize = cvSize(7, 7);
  CvSize shiftSize = cvSize(20, 20);
  CvSize max_range = cvSize(20, 20);
  int usePrevious = 0;

  // User values:
  if (lua_isnumber(L, 5)) {
    method = lua_tonumber(L, 5);
  }

  // HS only:
  if (lua_isnumber(L, 6)) {
    lagrangian = lua_tonumber(L, 6);
  }
  if (lua_isnumber(L, 7)) {
    iterations = lua_tonumber(L, 7);
  }

  // BM+LK only:
  if (lua_isnumber(L, 6) && lua_isnumber(L, 7)) {
    blockSize.width = lua_tonumber(L, 6);
    blockSize.height = lua_tonumber(L, 7);
  }
  if (lua_isnumber(L, 8) && lua_isnumber(L, 9)) {
    shiftSize.width = lua_tonumber(L, 8);
    shiftSize.height = lua_tonumber(L, 9);
  }
  if (lua_isnumber(L, 10) && lua_isnumber(L, 11)) {
    max_range.width = lua_tonumber(L, 10);
    max_range.height = lua_tonumber(L, 11);
  }
  if (lua_isnumber(L, 12)) {
    usePrevious = lua_tonumber(L, 12);
  }

  // Compute flow
  if (method == 1)
    {
      // Alloc outputs
      CvSize osize = cvSize((prev_ipl->width-blockSize.width)/shiftSize.width,
                            (prev_ipl->height-blockSize.height)/shiftSize.height);

      // Use previous results
      if (usePrevious == 1) {
        velx_ipl = libopencv_(Main_torchimg2opencv_32F)(velx);
        vely_ipl = libopencv_(Main_torchimg2opencv_32F)(vely);
      } else {
        velx_ipl = cvCreateImage(osize, IPL_DEPTH_32F, 1);
        vely_ipl = cvCreateImage(osize, IPL_DEPTH_32F, 1);
      }

      // Cv Call
      cvCalcOpticalFlowBM(prev_ipl, curr_ipl, blockSize, shiftSize,
                          max_range, usePrevious, velx_ipl, vely_ipl);
    }
  else if (method == 2)
    {
      // Alloc outputs
      CvSize osize = cvSize(prev_ipl->width, prev_ipl->height);

      velx_ipl = cvCreateImage(osize, IPL_DEPTH_32F, 1);
      vely_ipl = cvCreateImage(osize, IPL_DEPTH_32F, 1);

      // Cv Call
      cvCalcOpticalFlowLK(prev_ipl, curr_ipl, blockSize, velx_ipl, vely_ipl);
    }
  else if (method == 3)
    {
      // Alloc outputs
      CvSize osize = cvSize(prev_ipl->width, prev_ipl->height);

      // Use previous results
      if (usePrevious == 1) {
        velx_ipl = libopencv_(Main_torchimg2opencv_32F)(velx);
        vely_ipl = libopencv_(Main_torchimg2opencv_32F)(vely);
      } else {
        velx_ipl = cvCreateImage(osize, IPL_DEPTH_32F, 1);
        vely_ipl = cvCreateImage(osize, IPL_DEPTH_32F, 1);
      }

      // Iteration criterion
      CvTermCriteria term = cvTermCriteria(CV_TERMCRIT_ITER, iterations, 0);

      // Cv Call
      cvCalcOpticalFlowHS(prev_ipl, curr_ipl,
			  usePrevious, velx_ipl, vely_ipl,
                          lagrangian, term);
    }

  // return results
  libopencv_(Main_opencv32F2torch)(velx_ipl, velx);
  libopencv_(Main_opencv32F2torch)(vely_ipl, vely);

  // Deallocate IPL images
  cvReleaseImage(&prev_ipl);
  cvReleaseImage(&curr_ipl);
  cvReleaseImage(&vely_ipl);
  cvReleaseImage(&velx_ipl);

  return 0;
}

//============================================================
static int libopencv_(Main_cvGoodFeaturesToTrack) (lua_State *L) {
  // Get Tensor's Info
  THTensor * image     = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * points    = luaT_checkudata(L, 2, torch_(Tensor_id));

  CvSize dest_size         = cvSize(image->size[2], image->size[1]);
  IplImage * image_ipl     = libopencv_(Main_torchimg2opencv_8U)(image);

  IplImage * grey = cvCreateImage( dest_size, 8, 1 );

  cvCvtColor( image_ipl, grey, CV_BGR2GRAY );
  CvPoint2D32f* points_cv = 0;

  IplImage* eig =  cvCreateImage( dest_size, 32, 1 );
  IplImage* temp = cvCreateImage( dest_size, 32, 1 );

  int count = 500;
  double quality = 0.01;
  double min_distance = 10;
  int blocksize = 3;

  // User values:
  if (lua_isnumber(L, 3)) {
    count = lua_tonumber(L, 3);
  }
  if (lua_isnumber(L, 4)) {
    quality = lua_tonumber(L, 4);
  }
  if (lua_isnumber(L, 5)) {
    min_distance = lua_tonumber(L, 5);
  }
  if (lua_isnumber(L, 6)) {
    blocksize = lua_tonumber(L, 6);
  }

  points_cv = (CvPoint2D32f*)cvAlloc(count*sizeof(points_cv[0]));

  cvGoodFeaturesToTrack( grey, eig, temp, points_cv, &count,
			 quality, min_distance, NULL, blocksize, 0, 0.04 );

  // return results
  libopencv_(Main_opencvPoints2torch)(points_cv, count, points);

  // Deallocate points_cv
  cvFree(&points_cv);
  cvReleaseImage( &eig );
  cvReleaseImage( &temp );
  cvReleaseImage( &grey );
  cvReleaseImage( &image_ipl );

  return 0;
}

//============================================================
static int libopencv_(Main_cvCalcOpticalFlowPyrLK) (lua_State *L) {
  // Get Tensor's Info
  THTensor * image1 = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * image2 = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor * flow_x = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor * flow_y = luaT_checkudata(L, 4, torch_(Tensor_id));
  THTensor * points = luaT_checkudata(L, 5, torch_(Tensor_id));
  THTensor * image_out = luaT_checkudata(L, 6, torch_(Tensor_id));

  printf("Parsed args\n");
  int count = 500;
  double quality = 0.01;
  double min_distance = 10;
  int win_size = 10;

  // User values:
  if (lua_isnumber(L, 7)) {
    count = lua_tonumber(L, 7);
  }
  if (lua_isnumber(L, 8)) {
    quality = lua_tonumber(L, 8);
  }
  if (lua_isnumber(L, 9)) {
    min_distance = lua_tonumber(L, 9);
  }
  if (lua_isnumber(L, 10)) {
    win_size = lua_tonumber(L, 10);
  }
  printf("updated defaults\n");
  printf("size: (%ld,%ld)\n",image1->size[2], image1->size[1]);
  CvSize dest_size = cvSize(image1->size[2], image1->size[1]);
  IplImage * image1_ipl    = libopencv_(Main_torchimg2opencv_8U)(image1);
  IplImage * image2_ipl    = libopencv_(Main_torchimg2opencv_8U)(image2);
  THTensor_(resize3d)(image_out,
		      image1->size[0],image1->size[1],image1->size[2]);
  IplImage * image_out_ipl = libopencv_(Main_torchimg2opencv_8U)(image_out);
  printf("converted images\n");

  IplImage * grey1 = cvCreateImage( dest_size, 8, 1 );
  IplImage * grey2 = cvCreateImage( dest_size, 8, 1 );

  cvCvtColor( image1_ipl, grey1, CV_BGR2GRAY );
  cvCvtColor( image2_ipl, grey2, CV_BGR2GRAY );
  CvPoint2D32f* points1_cv = 0;
  CvPoint2D32f* points2_cv = 0;

  printf("Created IPL structures\n");
  IplImage* eig = cvCreateImage( dest_size, 32, 1 );
  IplImage* temp = cvCreateImage( dest_size, 32, 1 );

  // FIXME reuse points
  points1_cv = (CvPoint2D32f*)cvAlloc(count*sizeof(points1_cv[0]));
  points2_cv = (CvPoint2D32f*)cvAlloc(count*sizeof(points2_cv[0]));

  printf("Malloced points\n");
  cvGoodFeaturesToTrack( grey1, eig, temp, points1_cv, &count,
			 quality, min_distance, 0, 3, 0, 0.04 );
  printf("got good features for points1\n");
  /*
  cvFindCornerSubPix( grey1, points1_cv, count,
		      cvSize(win_size,win_size),
		      cvSize(-1,-1),
		      cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,
				     20,0.03));
  printf("Found SubPixel\n");
  */
  // Call Lucas Kanade algorithm
  char features_found[ count ];
  float feature_errors[ count ];
  CvSize pyr_sz = cvSize( image1_ipl->width+8, image1_ipl->height/3 );

  IplImage* pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
  IplImage* pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );

  cvCalcOpticalFlowPyrLK( grey1, grey2,
			  pyrA, pyrB,
			  points1_cv, points2_cv,
			  count,
			  cvSize( win_size, win_size ),
			  5, features_found, feature_errors,
			  cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );
  // make image
  int i;
  for( i = 0; i < count; i++ ) {
    if (features_found[i] >0){
      CvPoint p0 = cvPoint( cvRound( points1_cv[i].x),
			    cvRound( points1_cv[i].y));
      CvPoint p1 = cvPoint( cvRound( points2_cv[i].x),
			    cvRound( points2_cv[i].y));
      cvLine( image_out_ipl, p0, p1, CV_RGB(255,0,0), 1, CV_AA, 0);
      //create the flow vectors to be compatible with the other
      //opticalFlows
      if (((p1.x > 0) && (p1.x < flow_x->size[0])) &&
	  ((p1.y > 0) && (p1.y < flow_x->size[1]))) {
	THTensor_(set2d)(flow_x,p1.x,p1.y,points1_cv[i].x - points2_cv[i].x);
	THTensor_(set2d)(flow_y,p1.x,p1.y,points1_cv[i].y - points2_cv[i].y);
      }
    }
  }

  // return results
  libopencv_(Main_opencvPoints2torch)(points2_cv, count, points);
  libopencv_(Main_opencv8U2torch)(image_out_ipl, image_out);

  // Deallocate points_cv
  cvFree(&points1_cv);
  cvFree(&points2_cv);
  cvReleaseImage( &eig );
  cvReleaseImage( &temp );
  cvReleaseImage( &pyrA );
  cvReleaseImage( &pyrB );
  cvReleaseImage( &grey1 );
  cvReleaseImage( &grey2);
  cvReleaseImage( &image1_ipl );
  cvReleaseImage( &image2_ipl );
  cvReleaseImage( &image_out_ipl );

  return 0;
}

//============================================================
static int libopencv_(Main_cvTrackPyrLK) (lua_State *L) {
  // Get Tensor's Info
  THTensor * image1  = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * image2  = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor * points1 = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor * points2 = luaT_checkudata(L, 4, torch_(Tensor_id));
  THTensor * ff = 0;
  THTensor * fe = 0;

  int count = points1->size[0];
  int win_size = 10;

  // User values:
  if (lua_isnumber(L, 5)) {
    win_size = lua_tonumber(L, 5);
  }

  if (!lua_isnil(L,6)) {
    ff = luaT_checkudata(L,6,torch_(Tensor_id));
    THTensor_(resize1d)(ff,count);
  }
  if (!lua_isnil(L,7)) {
    fe = luaT_checkudata(L,7,torch_(Tensor_id));
    THTensor_(resize1d)(fe,count);
  }

  CvSize dest_size = cvSize(image1->size[2], image1->size[1]);
  IplImage * image1_ipl = libopencv_(Main_torchimg2opencv_8U)(image1);
  IplImage * image2_ipl = libopencv_(Main_torchimg2opencv_8U)(image2);


  IplImage * grey1 = cvCreateImage( dest_size, 8, 1 );
  IplImage * grey2 = cvCreateImage( dest_size, 8, 1 );

  cvCvtColor( image1_ipl, grey1, CV_BGR2GRAY );
  cvCvtColor( image2_ipl, grey2, CV_BGR2GRAY );
  CvPoint2D32f* points1_cv = libopencv_(Main_torch2opencvPoints)(points1);
  CvPoint2D32f* points2_cv = 0;
  points2_cv = (CvPoint2D32f*)cvAlloc(count*sizeof(points2_cv[0]));


  // Call Lucas Kanade algorithm
  char features_found[ count ];
  float feature_errors[ count ];
  CvSize pyr_sz = cvSize( image1_ipl->width+8, image1_ipl->height/3 );

  IplImage* pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
  IplImage* pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );

  cvCalcOpticalFlowPyrLK( grey1, grey2,
			  pyrA, pyrB,
			  points1_cv, points2_cv,
			  count,
			  cvSize( win_size, win_size ),
			  5, features_found, feature_errors,
			  cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );

  // return results
  libopencv_(Main_opencvPoints2torch)(points2_cv, count, points2);
  int i;
  if (ff != 0){
    for(i=0;i<count;i++){
      THTensor_(set1d)(ff,i,features_found[i]);
    }
  }
  if (fe != 0){
    for(i=0;i<count;i++){
      THTensor_(set1d)(fe,i,feature_errors[i]);
    }
  }
  // Deallocate points_cv
  cvFree(&points1_cv);
  cvFree(&points2_cv);
  cvReleaseImage( &pyrA );
  cvReleaseImage( &pyrB );
  cvReleaseImage( &grey1 );
  cvReleaseImage( &grey2);
  cvReleaseImage( &image1_ipl );
  cvReleaseImage( &image2_ipl );

  return 0;
}

//============================================================
// draws red flow lines on an image (for visualizing the flow)
static int libopencv_(Main_cvDrawFlowlinesOnImage) (lua_State *L) {
  THTensor * points1 = luaT_checkudata(L,1, torch_(Tensor_id));
  THTensor * points2 = luaT_checkudata(L,2, torch_(Tensor_id));
  THTensor * image   = luaT_checkudata(L,3, torch_(Tensor_id));
  THTensor * color   = luaT_checkudata(L,4, torch_(Tensor_id));
  THTensor * mask    = NULL;
  int usemask = 0;
  if (!lua_isnil(L,5)){
    usemask = 1;
    mask = luaT_checkudata(L,5, torch_(Tensor_id));
  }
  IplImage * image_ipl = libopencv_(Main_torchimg2opencv_8U)(image);
  CvScalar color_cv = CV_RGB(THTensor_(get1d)(color,0),
			     THTensor_(get1d)(color,1),
			     THTensor_(get1d)(color,2));
  int count = points1->size[0];
  int i;
  for( i = 0; i < count; i++ ) {
    if ( !usemask || (THTensor_(get1d)(mask,i) > 0)){
      CvPoint p0 = cvPoint( cvRound( THTensor_(get2d)(points1,i,0)),
			    cvRound( THTensor_(get2d)(points1,i,1)));
      CvPoint p1 = cvPoint( cvRound( THTensor_(get2d)(points2,i,0)),
			    cvRound( THTensor_(get2d)(points2,i,1)));
      cvLine( image_ipl, p0, p1, color_cv, 2, CV_AA, 0);
    }
  }
  // return results
  libopencv_(Main_opencv8U2torch)(image_ipl, image);
  cvReleaseImage( &image_ipl );
  return 0;
}
/*
 * to create a smooth flow map from the dense tracking:
 *  -- compute voronoi tessalation around sparse input points
 *  -- interpolate to fill each triangle
 *  -- return dense field
 */
static int libopencv_(Main_smoothVoronoi) (lua_State *L) {
  THTensor * points = luaT_checkudata(L,1, torch_(Tensor_id));
  THTensor * data   = luaT_checkudata(L,2, torch_(Tensor_id));
  THTensor * output = luaT_checkudata(L,3, torch_(Tensor_id));
  real * output_pt[8];
  int i;
  output_pt[0] = THTensor_(data)(output);

  for (i=1;i<output->size[0];i++){
    output_pt[i] = output_pt[0] + i*output->stride[0];
  }
  /* annoying set this higher if you get errors about points being out
     of range */
  int ex = 1000;
  int w = 2 * ex + output->size[2];
  int h = 2 * ex + output->size[1];
  CvRect rect = { -ex, -ex, w ,h };
  CvMemStorage* storage;
  CvSubdiv2D* subdiv;

  storage = cvCreateMemStorage(points->size[0]*sizeof(CvSubdiv2DPoint));
  subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv),
                             sizeof(CvSubdiv2DPoint),
                             sizeof(CvQuadEdge2D),
                             storage );
  cvInitSubdivDelaunay2D( subdiv, rect );

  int count = points->size[0];
  for( i = 0; i < count; i++ ) {
    CvPoint2D32f fp = cvPoint2D32f((double)THTensor_(get2d)(points,i,0),
                                   (double)THTensor_(get2d)(points,i,1));
    CvSubdiv2DPoint * e = cvSubdivDelaunay2DInsert( subdiv, fp );
    e->flags = i; /* store the index of the point */
  }


  cvCalcSubdivVoronoi2D( subdiv );

  /*
   * now loop through the image and for each point find the triangle
   * of points around it and interpolate
   */
  int x,y;
  CvPoint2D32f fp;
  CvSubdiv2DEdge e = 0;
  CvSubdiv2DEdge e0 = 0;
  CvSubdiv2DPoint* p = NULL;
  CvSubdiv2DPoint* org = NULL;
  real data_w[3][8];
  real data_x[3];
  real data_y[3];

  for (y=0;y<output->size[1];y++){
    for (x=0;x<output->size[2];x++){
      fp = cvPoint2D32f((double)x, (double)y);
      count = 0;
      /* find first point on a triangle around this point */
      cvSubdiv2DLocate( subdiv, fp, &e0, &p );
      if( e0 ) {
        e = e0;
        do // Always 3 edges -- this is a triangulation, after all.
          {
            // Do something with e ...
            e   = cvSubdiv2DGetEdge(e,CV_NEXT_AROUND_LEFT);
            org = cvSubdiv2DEdgeOrg(e);
            data_x[count] = org->pt.x;
            data_y[count] = org->pt.y;
            for(i=0;i<data->size[1];i++){
              data_w[count][i] = THTensor_(get2d)(data,org->flags,i);
            }
            count++;
          }
        while( e != e0 );
        /* interpolate weights from 3 points */
        /* determinant of the original position matrix */
        real DET =
          data_x[0]*data_y[1] -
          data_x[1]*data_y[0] +
          data_x[1]*data_y[2] -
          data_x[2]*data_y[1] +
          data_x[2]*data_y[0] -
          data_x[0]*data_y[2];
        real A,B,C;

        for (i=0;i<output->size[0];i++){
          A = ((data_y[1]-data_y[2])*data_w[0][i] +
               (data_y[2]-data_y[0])*data_w[1][i] +
               (data_y[0]-data_y[1])*data_w[2][i]) / DET ;
          B = ((data_x[2]-data_x[1])*data_w[0][i] +
               (data_x[0]-data_x[2])*data_w[1][i] +
               (data_x[1]-data_x[0])*data_w[2][i]) / DET ;
          C = ((data_x[1]*data_y[2]-data_x[2]*data_y[1])*data_w[0][i] +
               (data_x[2]*data_y[0]-data_x[0]*data_y[2])*data_w[1][i] +
               (data_x[0]*data_y[1]-data_x[1]*data_y[0])*data_w[2][i]) /
            DET ;
          THTensor_(set3d)(output,i,y,x,A*x+B*y+C);
        }
        //printf("  count: %d\n",count);
        count = 0;
      }
    }
  }
  cvReleaseMemStorage( &storage );

  return 0;
}

//============================================================
static int libopencv_(Main_cvCanny) (lua_State *L) {
  // Get Tensor's Info
  THTensor * source = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * dest   = luaT_checkudata(L, 2, torch_(Tensor_id));

  /* first convert the tensors into proper opencv Canny format */

  // Pointers
  uchar * source_data;
  int channels = source->size[0];
  int source_step;
  CvSize source_size = cvSize(source->size[2], source->size[1]);

  // Create ipl image
  IplImage * source_ipl = cvCreateImage(source_size, IPL_DEPTH_8U, channels);

  // get pointer to raw data
  cvGetRawData(source_ipl, (uchar**)&source_data, &source_step, &source_size);

  // copy
  THTensor *tensor = THTensor_(newContiguous)(source);
  int i = 0;
  for (i=0;i<source->size[1];i++){
    uchar * sourcep = source_data + source_step*i;
    THTensor *tslice = THTensor_(newSelect)(tensor,1,i);
    // copy
    TH_TENSOR_APPLY(real, tslice,
                    *sourcep = (uchar)(*tslice_data * 255.0);
                    sourcep++;
		    );
    THTensor_(free)(tslice);
  }
  // free
  THTensor_(free)(tensor);

  IplImage * dest_ipl = cvCreateImage(cvGetSize(source_ipl), IPL_DEPTH_8U,
                                      source_ipl->nChannels);

  /* read the params values */
  // Thresholds with default values
  double low_threshold = 50;
  double high_threshold = 150;
  int blur_size = 3;
  int aperture_size = 3;
  if (lua_isnumber(L, 3)) low_threshold = lua_tonumber(L, 3);
  if (lua_isnumber(L, 4)) high_threshold = lua_tonumber(L, 4);
  if (lua_isnumber(L, 5)) blur_size = lua_tonumber(L, 5);
  if (lua_isnumber(L, 6)) aperture_size = lua_tonumber(L, 6);
  // if we have a percent argument we need to find the thresholds
  if (lua_isnumber(L, 7))
    { // here we compute the sobel filtering and its histogram
      // to assess where the thresholds should be
      // this code is inspired of the matlab method
      double percent = lua_tonumber(L, 7);
      double vmin,vmax;
      CvSize size = cvGetSize(source_ipl);
      IplImage* mag = cvCreateImage(size, IPL_DEPTH_32F,1);
      cvSetZero(mag);
      IplImage* drv = cvCreateImage(size, IPL_DEPTH_16S,1);
      IplImage* drv32f = cvCreateImage(size, IPL_DEPTH_32F,1);
      cvSobel(source_ipl, drv, 1 , 0, aperture_size);
      cvConvertScale(drv,drv32f,1,0);
      cvSquareAcc(drv32f,mag,NULL);
      cvSobel(source_ipl, drv, 0 , 1, aperture_size);
      cvConvertScale(drv,drv32f,1,0);
      cvSquareAcc(drv32f,mag,NULL);
      cvbSqrt((float*)(mag->imageData),(float*)(mag->imageData), mag->imageSize/sizeof(float));
      cvReleaseImage(&drv);
      cvReleaseImage(&drv32f);
      // compute histogram
      #define NB_BINS 64
      int nbBins = NB_BINS;
      CvHistogram *hist;
      cvMinMaxLoc(mag,&vmin,&vmax,NULL,NULL,NULL);
      float hranges_arr[] = {(float)vmin,(float)vmax};
      float* hranges = hranges_arr;
      hist = cvCreateHist(1,&nbBins,CV_HIST_ARRAY,&hranges,1);
      cvCalcHist(&mag,hist,0,NULL);
      cvReleaseImage(&mag);
      CvMat mat;
      cvGetMat(hist->bins, &mat, 0 , 1);
      double binStep = (vmax-vmin)/NB_BINS;
      double qty = 100;
      double nbelmts = 0;
      int idx=0;
      double tot = (size.height*size.width);
      while (qty > percent && idx < mat.rows)
        {
          nbelmts += cvmGet(&mat,idx,0);
          qty = (tot-nbelmts)*100/tot;
          idx++;
        }
      high_threshold = (double)idx*binStep;
      low_threshold = 0.4*high_threshold;
    }

  /* now gaussian smooth to reduce the noise */
  if (blur_size > 1)
    {
      cvSmooth(source_ipl, dest_ipl, CV_GAUSSIAN, blur_size, blur_size, 0, 0);
      // Simple call to CV function
      cvCanny(dest_ipl, dest_ipl, low_threshold, high_threshold, aperture_size);
    }
  else
    {
      // Simple call to CV function
      cvCanny(source_ipl, dest_ipl, low_threshold, high_threshold, aperture_size);
    }

  /* there was a bug in converting the result back in torch format */
  /* so I wrote my own */
  CvMat dststub, * dmat =  cvGetMat( dest_ipl, &dststub, NULL, 0 );
  THTensor_(resize2d)(dest, dmat->rows, dmat->cols);
  tensor = THTensor_(newContiguous)(dest);
  for(i = 0; i < dmat->rows; i++ )
    {
      const uchar* dmat_p =  dmat->data.ptr + dmat->step*i;
      THTensor *tslice = THTensor_(newSelect)(tensor,0,i);
      // copy
      TH_TENSOR_APPLY(real, tslice,
                      *tslice_data = ((real)(*dmat_p))/255.0;
                      dmat_p++;
                      );
      THTensor_(free)(tslice);
    }
  THTensor_(free)(tensor);

  // Deallocate headers
  cvReleaseImageHeader(&source_ipl);
  cvReleaseImageHeader(&dest_ipl);

  // return the thresholds used for the computation
  lua_pushnumber(L, low_threshold);
  lua_pushnumber(L, high_threshold);
  return 2;
}

//============================================================
static int libopencv_(Main_cvEqualizeHist) (lua_State *L) {
  // Get Tensor's Info
  THTensor * source = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * dest   = luaT_checkudata(L, 2, torch_(Tensor_id));

  // Generate IPL headers
  IplImage * source_ipl = libopencv_(Main_torchimg2opencv_8U)(source);
  IplImage * dest_ipl = cvCreateImage(cvGetSize(source_ipl), IPL_DEPTH_8U,
                                      source_ipl->nChannels);


  // Simple call to CV function
  cvEqualizeHist(source_ipl, dest_ipl);

  // return results
  libopencv_(Main_opencv8U2torch)(dest_ipl, dest);

  // Deallocate headers
  cvReleaseImageHeader(&source_ipl);
  cvReleaseImageHeader(&dest_ipl);

  return 0;
}

//============================================================
static int libopencv_(Main_cvWarpAffine) (lua_State *L) {
  // Get Tensor's Info
  THTensor * source = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * dest   = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor * warp   = luaT_checkudata(L, 3, torch_(Tensor_id));

  THArgCheck(warp->size[0] == 2 , 1, "warp matrix: 2x3 Tensor expected");
  THArgCheck(warp->size[1] == 3 , 1, "warp matrix: 2x3 Tensor expected");

  // Generate IPL headers

  IplImage * source_ipl = libopencv_(Main_torchimg2opencv_8U)(source);
  IplImage * dest_ipl = cvCreateImage(cvGetSize(source_ipl), IPL_DEPTH_8U,
                                      source_ipl->nChannels);
  CvMat* warp_mat = cvCreateMat(2,3,CV_32FC1);

  // Copy warp transformation matrix
  THTensor *tensor = THTensor_(newContiguous)(warp);
  float* ptr = warp_mat->data.fl;
  TH_TENSOR_APPLY(real, tensor,
                  *ptr = (float)*tensor_data;
                  ptr++;
                  );
  THTensor_(free)(tensor);

  // Simple call to CV function
  cvWarpAffine(source_ipl, dest_ipl, warp_mat,
               CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0));

  // Return results
  libopencv_(Main_opencv8U2torch)(dest_ipl, dest);

  // Deallocate headers
  cvReleaseImageHeader(&source_ipl);
  cvReleaseImageHeader(&dest_ipl);
  cvReleaseMat( &warp_mat );

  return 0;
}

//============================================================
static int libopencv_(Main_cvGetAffineTransform) (lua_State *L) {
  // Get Tensor's Info
  THTensor * src = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor * dst   = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor * warp   = luaT_checkudata(L, 3, torch_(Tensor_id));

  /* THArgCheck(src->size[0] == 3 , 1, "input: 2x3 Tensor expected"); */
  /* THArgCheck(src->size[1] == 2 , 1, "input: 2x3 Tensor expected"); */
  /* THArgCheck(dst->size[0] == 3 , 2, "input: 2x3 Tensor expected"); */
  /* THArgCheck(dst->size[1] == 2 , 2, "input: 2x3 Tensor expected"); */

  CvPoint2D32f* srcTri = libopencv_(Main_torch2opencvPoints)(src);
  CvPoint2D32f* dstTri = libopencv_(Main_torch2opencvPoints)(dst);

  CvMat* warp_mat = cvCreateMat(2,3,CV_32FC1);
  cvGetAffineTransform( srcTri, dstTri, warp_mat );

  libopencv_(Main_opencvMat2torch)(warp_mat,warp);

  // Deallocate headers
  cvFree(&srcTri);
  cvFree(&dstTri);
  cvReleaseMat(&warp_mat);

  return 0;
}


/*
 * compute fundamental matrix from matching points between 2 images
 */
static int libopencv_(Main_cvFindFundamental) (lua_State *L) {
  THTensor * points1_th      = luaT_checkudata(L,1, torch_(Tensor_id));
  THTensor * points2_th      = luaT_checkudata(L,2, torch_(Tensor_id));
  THTensor * fundamental_th  = luaT_checkudata(L,3, torch_(Tensor_id));
  THTensor * status_th       = luaT_checkudata(L,4, torch_(Tensor_id));

  THTensor_(resize2d)(fundamental_th,3,3);

  real * points1_pt = THTensor_(data)(points1_th);
  real * points2_pt = THTensor_(data)(points2_th);
  real * status_pt = THTensor_(data)(status_th);

  int    numPoints  = points1_th->size[0];

  CvMat* points1 = cvCreateMat(2,numPoints,CV_32F);
  CvMat* points2 = cvCreateMat(2,numPoints,CV_32F);

  int i;
  for ( i = 0; i < numPoints; i++) {
    cvSetReal2D(points1,0,i,(float)*points1_pt++);
    cvSetReal2D(points1,1,i,(float)*points1_pt++);

    cvSetReal2D(points2,0,i,(float)*points2_pt++);
    cvSetReal2D(points2,1,i,(float)*points2_pt++);
  }

  CvMat* status  = cvCreateMat(1,numPoints,CV_8U);
  CvMat* fundamentalMatrix = cvCreateMat(3,3,CV_32F);
  int    method = CV_FM_RANSAC;
  double param1 = 1.;
  double param2 = 0.99;

  cvFindFundamentalMat(points1, points2, fundamentalMatrix,
                       method,param1,param2,status);

  int j;
  for ( i = 0; i < 3; i++) {
    for ( j = 0; j < 3; j++) {
      THTensor_(set2d)(fundamental_th,i,j,
                       (real)cvGetReal2D(fundamentalMatrix,i,j));
    }
  }
  for ( i = 0; i < numPoints; i++) {
    *status_pt++ = cvGetReal2D(status,0,i);
  }
  //cleanup
  cvReleaseMat(&status);
  cvReleaseMat(&fundamentalMatrix);
  cvReleaseMat(&points1);
  cvReleaseMat(&points2);
  return 0;
}


//============================================================
// Register functions in LUA
//
static const luaL_reg libopencv_(Main__) [] =
{
  {"FindFundamental",      libopencv_(Main_cvFindFundamental)},
  {"GetAffineTransform",   libopencv_(Main_cvGetAffineTransform)},
  {"WarpAffine",           libopencv_(Main_cvWarpAffine)},
  {"EqualizeHist",         libopencv_(Main_cvEqualizeHist)},
  {"Canny",                libopencv_(Main_cvCanny)},
  {"smoothVoronoi",        libopencv_(Main_smoothVoronoi)},
  {"drawFlowlinesOnImage", libopencv_(Main_cvDrawFlowlinesOnImage)},
  {"TrackPyrLK",           libopencv_(Main_cvTrackPyrLK)},
  {"CalcOpticalFlowPyrLK", libopencv_(Main_cvCalcOpticalFlowPyrLK)},
  {"CalcOpticalFlow",      libopencv_(Main_cvCalcOpticalFlow)},
  {"CornerHarris",         libopencv_(Main_cvCornerHarris)},
  {"GoodFeaturesToTrack",  libopencv_(Main_cvGoodFeaturesToTrack)},
  {NULL, NULL}  /* sentinel */
};

DLL_EXPORT int libopencv_(Main_init) (lua_State *L) {
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, libopencv_(Main__), "libopencv");
  return 1;
}

#endif
