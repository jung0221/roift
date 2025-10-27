#ifndef _GFT_COMPAT_H_
#define _GFT_COMPAT_H_

#include "gft_scene32.h"
#include "gft_filtering3.h"
#include "gft_gradient3.h"

namespace gft {
  namespace Scene32 {
    // Provide legacy names used elsewhere that map to existing functions.
    inline sScene32 *FastGaussianBlur3(sScene32 *scn){
      // No FastGaussianBlur currently exported; call GaussianBlur
      return gft::Scene32::GaussianBlur(scn);
    }

    /* Some sources call FastGaussianBlur (without suffix); provide alias */
    inline sScene32 *FastGaussianBlur(sScene32 *scn){
      return gft::Scene32::GaussianBlur(scn);
    }

    inline sScene32 *Subsampling3(sScene32 *scn){
      return gft::Scene32::Subsampling(scn);
    }

    inline sScene32 *FastLinearInterpCentr3(sScene32 *scn, float dx, float dy, float dz){
      return gft::Scene32::LinearInterp(scn, dx, dy, dz);
    }
  }

  namespace Gradient3 {
    inline sGradient3 *ReadCompressed(const char *fn){
      // Provide a fallback to Read (compressed I/O depends on zlib availability)
      return Read((char *)fn);
    }
    inline void WriteCompressed(sGradient3 *g, const char *fn){
      Write(g, (char *)fn);
    }
  }
}

#endif
