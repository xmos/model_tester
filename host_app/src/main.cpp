#include "model.tflite.h"
#include <stdio.h>
#include <string.h>
#include "image.h"

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
  
  // Set input
  int8_t *in = model_input(0)->data.int8;
  int size = model_input_size(0);
  for (int i=0;i<size;++i) {
    in[i] = lion[i]-128;
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

  return 0;
}