#include <platform.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <xcore/channel.h>
#include "model.tflite.h"

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

void init(unsigned flash_data) { model_init((void *)flash_data); }

void run() {
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
}

extern "C" {
void model_init(unsigned flash_data) { init(flash_data); }

void inference() { run(); }
}
