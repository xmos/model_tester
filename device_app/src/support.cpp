#include <platform.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <xcore/channel.h>
#include "model.tflite.h"
#include "image.h"


// The sample input image is initialized at the beginning of the tensor arena.
// Before we run inference, the input image is copied to the input tensor
// location in the tensor arena.
// With this optimization, we don't need an extra array to store the input
// image.
uint8_t tensor_arena[LARGEST_TENSOR_ARENA_SIZE] __attribute__((aligned(8))) = IMAGE;


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
  // Set input
  int8_t *in = model_input(0)->data.int8;
  int size = model_input_size(0);
  for (int i=0;i<size;++i) {
    in[i] = tensor_arena[i] - 128;
  }

  model_invoke();

  // Find top three classes
  int maxIndex1 = -1;
  int max1 = -128;
  int maxIndex2 = -1;
  int max2 = -128;
  int maxIndex3 = -1;
  int max3 = -128;
  int8_t *out = model_output(0)->data.int8;
  for (int i = 0; i < model_output_size(0); ++i) {
    if (out[i] > max1) {
      max3 = max2;
      maxIndex3 = maxIndex2;
      max2 = max1;
      maxIndex2 = maxIndex1;
      max1 = out[i];
      maxIndex1 = i;
    }
  }

  printf("\nClass with max1 value = %d and probability = %f", maxIndex1, dequantize_output(max1));
  printf("\nClass with max2 value = %d and probability = %f", maxIndex2, dequantize_output(max2));
  printf("\nClass with max3 value = %d and probability = %f", maxIndex3, dequantize_output(max3));
}

extern "C" {
void model_init(unsigned flash_data) { init(flash_data); }

void inference() { run(); }
}
