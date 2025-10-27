
#include "gft_image8.h"

namespace gft{
  namespace Image8{

    sImage8 *Create(int ncols, int nrows){
      sImage8 *img = NULL;
      int i;
      
      img = (sImage8 *)malloc(sizeof(sImage8));
      if(img == NULL){
	printf("Error: Insufficient memory.\n");
	exit(1);
      }
      img->nrows = nrows;
      img->ncols = ncols;
      img->n = nrows*ncols;
      
      img->data = (uchar *)calloc(ncols*nrows, sizeof(uchar));
      if(img->data == NULL){
	printf("Error: Insufficient memory.\n");
	exit(1);
      }

      img->array = (uchar**)malloc(nrows*sizeof(uchar *));
      if(img->array == NULL){
	printf("Error: Insufficient memory.\n");
	exit(1);
      }
      for(i = 0; i < nrows; i++){
	img->array[i] = (img->data + i*ncols);
      }
      return img;
    }
    

    void Destroy(sImage8 **img){
      sImage8 *tmp;
      if(img!=NULL){
	tmp = *img;
	if(tmp != NULL){
	  if(tmp->data  != NULL) free(tmp->data);
	  if(tmp->array != NULL) free(tmp->array);
	  free(tmp);
	  *img = NULL;
	}
      }
    }




    void Write(sImage8 *img,char *filename){
      FILE *fp;
      int i, n, Imax;
      
      fp = fopen(filename,"wb");
      if (fp == NULL){
        fprintf(stderr,"Cannot open %s\n",filename);
        exit(-1);
      }
      n = img->ncols*img->nrows;
      Imax = GetMaxVal(img);

      fprintf(fp,"P2\n");
      fprintf(fp,"%d %d\n",img->ncols,img->nrows);
      if (Imax==0) Imax++;
      fprintf(fp,"%d\n",Imax);
      
      for (i=0; i < n; i++){
	if (img->data[i]==INT_MAX)
	  fprintf(fp,"%d ",Imax+1);
        else
	  fprintf(fp,"%d ",img->data[i]);
	if (((i+1)%17) == 0)
	  fprintf(fp,"\n");
      }
      fclose(fp);
    }
    

    int GetMaxVal(sImage8 *img){
      int i,max,n;
      n = img->ncols*img->nrows;
      max = img->data[0];
      for (i=1; i < n; i++)
        if (img->data[i] > max)
	  max = img->data[i];
      //img->maxval = MAX(max, img->maxval);
      return(max);
    }
    

  } /*end Image8 namespace*/
} /*end gft namespace*/

