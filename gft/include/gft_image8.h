#ifndef _GFT_IMAGE8_H_
#define _GFT_IMAGE8_H_

#include "gft_common.h"
#include "gft_image32.h"

namespace gft{
  namespace Image8{

    struct sImage8 {
      uchar *data;
      uchar **array;
      int nrows; /* numero de linhas (altura) */
      int ncols; /* numero de colunas (largura) */
      int n;     /* numero de pixels */
    };


    sImage8 *Create(int ncols, int nrows);
    void     Destroy(sImage8 **img);

    /**
     * \brief Writes an image to a file on the disc in PGM format. 
     */
    void     Write(sImage8 *img, char *filename);

    /**
     * \brief Returns the maximum pixel value present in a given image.
     */
    int     GetMaxVal(sImage8 *img);
    
  } /*end Image8 namespace*/

  typedef Image8::sImage8 sImage8;

} /*end gft namespace*/

#endif

