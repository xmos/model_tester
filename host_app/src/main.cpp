#include "model.tflite.h"
#include <stdio.h>
#include <string.h>

#define MAX_WEIGHT_PARAMS_SIZE 5000000
int load_binary_file(const char *filename, uint32_t *content,
                            size_t size) {
  FILE *fd = fopen(filename, "rb");
  if (fd == NULL) {
    fprintf(stderr, "Cannot read model/param file %s\n", filename);
    exit(1);
  }
  int s = fread(content, 1, size, fd);
  fclose(fd);

  return s;
}
uint32_t weight_params[MAX_WEIGHT_PARAMS_SIZE];

// Simple checksum calc
unsigned char checksum_calc(char *data, unsigned int length)
{
  static char sum;
  static char * end;
  sum = 0;
  end = data + length;

  do
  {
      sum -= *data++;
  } while (data != end);
  return sum;
}

// Quantize float to int8
int quantize_input(float n) {
  return n/model_input_scale(0) + model_input_zeropoint(0);
}

// Dequantize int8 to float
float dequantize_output(int n) {
  return (n - model_output_zeropoint(0)) * model_output_scale(0);
}


int main(int argc, char *argv[])
{
  uint32_t *weights_ptr = nullptr;
  if(argc > 1) {
    weights_ptr = weight_params;
    (void)load_binary_file(argv[1], weight_params, MAX_WEIGHT_PARAMS_SIZE);
  }

  // Initialize with weights file if provided
  if(model_init(weights_ptr)) {
    printf("Error!\n");
  }
  
  // Set inputs
  for(int n=0; n< model_inputs(); ++n) {
    int8_t *in = model_input(n)->data.int8;
    int size = model_input_size(n);
    int k = -128;
    // Create input data as 
    // -128, -125, -122, ..., 127, -128, -125, ...
    for (int i=0;i<size;++i) {
      if (k >= 128) {
        k = -128;
      }
      in[i] = k;
      k = k + 3;
    }
  }

  // Run inference
  model_invoke();

  // Print outputs
  for(int n=0; n< model_outputs(); ++n) {
    int8_t *out = model_output(n)->data.int8;
    int size = model_output_size(n);
    printf("Output %d\n", n);
    for (int i=0;i<size;++i){
      printf("%d,",(int)out[i]);
    }
    printf("\nchecksum : %d\n\n", (int)checksum_calc((char*)out, model_output_size(n)));
  }

  return 0;
}