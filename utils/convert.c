#include <stdio.h>

int ConvertModelFile(const char *in_name, const char *out_name) {
  FILE *fp = fopen(in_name, "rb"), *wfp = fopen(out_name, "wb");
  if (!fp) return -1;

  const int kMagic = 0x1909eca1;
  int temp;
  fwrite(&kMagic, sizeof(int), 1, wfp);
  temp = 7;
  fwrite(&temp, sizeof(int), 1, wfp);
  temp = 0;
  fwrite(&temp, sizeof(int), 1, wfp);
  temp = 0;
  fwrite(&temp, sizeof(int), 1, wfp);

  int len_weight_C1;
  int len_bias_C1;
  int len_weight_S2;
  int len_bias_S2;
  int len_weight_C3;
  int len_bias_C3;
  int len_weight_S4;
  int len_bias_S4;
  int len_weight_C5;
  int len_bias_C5;
  int len_weight_output;
  int len_bias_output;
  float farr[50000];

  fseek(fp, 25 * sizeof(int), SEEK_CUR);

  fread(&len_weight_C1, sizeof(int), 1, fp);
  printf("len_weight_C1: %i\n", len_weight_C1);
  fread(&len_bias_C1, sizeof(int), 1, fp);
  printf("len_bias_C1: %i\n", len_bias_C1);
  fread(&len_weight_S2, sizeof(int), 1, fp);
  printf("len_weight_S2: %i\n", len_weight_S2);
  fread(&len_bias_S2, sizeof(int), 1, fp);
  printf("len_bias_S2: %i\n", len_bias_S2);
  fread(&len_weight_C3, sizeof(int), 1, fp);
  printf("len_weight_C3: %i\n", len_weight_C3);
  fread(&len_bias_C3, sizeof(int), 1, fp);
  printf("len_bias_C3: %i\n", len_bias_C3);
  fread(&len_weight_S4, sizeof(int), 1, fp);
  printf("len_weight_S4: %i\n", len_weight_S4);
  fread(&len_bias_S4, sizeof(int), 1, fp);
  printf("len_bias_S4: %i\n", len_bias_S4);
  fread(&len_weight_C5, sizeof(int), 1, fp);
  printf("len_weight_C5: %i\n", len_weight_C5);
  fread(&len_bias_C5, sizeof(int), 1, fp);
  printf("len_bias_C5: %i\n", len_bias_C5);
  fread(&len_weight_output, sizeof(int), 1, fp);
  printf("len_weight_output: %i\n", len_weight_output);
  fread(&len_bias_output, sizeof(int), 1, fp);
  printf("len_bias_output: %i\n", len_bias_output);

  fseek(fp, 7 * sizeof(int), SEEK_CUR);

  fwrite(&len_weight_C1, sizeof(int), 1, wfp);
  fwrite(&len_bias_C1, sizeof(int), 1, wfp);
  fread(farr, len_weight_C1 * sizeof(float), 1, fp);
  fwrite(farr, len_weight_C1 * sizeof(float), 1, wfp);
  fread(farr, len_bias_C1 * sizeof(float), 1, fp);
  for (int i = 0; i < len_bias_C1; ++i) {
    printf("%f ", farr[i]);
  }
  printf("\n");
  fwrite(farr, len_bias_C1 * sizeof(float), 1, wfp);

  fwrite(&len_weight_S2, sizeof(int), 1, wfp);
  fwrite(&len_bias_S2, sizeof(int), 1, wfp);
  fread(farr, len_weight_S2 * sizeof(float), 1, fp);
  fwrite(farr, len_weight_S2 * sizeof(float), 1, wfp);
  fread(farr, len_bias_S2 * sizeof(float), 1, fp);
  fwrite(farr, len_bias_S2 * sizeof(float), 1, wfp);

  fwrite(&len_weight_C3, sizeof(int), 1, wfp);
  fwrite(&len_bias_C3, sizeof(int), 1, wfp);
  fread(farr, len_weight_C3 * sizeof(float), 1, fp);
  fwrite(farr, len_weight_C3 * sizeof(float), 1, wfp);
  fread(farr, len_bias_C3 * sizeof(float), 1, fp);
  fwrite(farr, len_bias_C3 * sizeof(float), 1, wfp);

  fwrite(&len_weight_S4, sizeof(int), 1, wfp);
  fwrite(&len_bias_S4, sizeof(int), 1, wfp);
  fread(farr, len_weight_S4 * sizeof(float), 1, fp);
  fwrite(farr, len_weight_S4 * sizeof(float), 1, wfp);
  fread(farr, len_bias_S4 * sizeof(float), 1, fp);
  fwrite(farr, len_bias_S4 * sizeof(float), 1, wfp);

  fwrite(&len_weight_C5, sizeof(int), 1, wfp);
  fwrite(&len_bias_C5, sizeof(int), 1, wfp);
  fread(farr, len_weight_C5 * sizeof(float), 1, fp);
  fwrite(farr, len_weight_C5 * sizeof(float), 1, wfp);
  fread(farr, len_bias_C5 * sizeof(float), 1, fp);
  fwrite(farr, len_bias_C5 * sizeof(float), 1, wfp);

  fwrite(&len_weight_output, sizeof(int), 1, wfp);
  fwrite(&len_bias_output, sizeof(int), 1, wfp);
  fread(farr, len_weight_output * sizeof(float), 1, fp);
  fwrite(farr, len_weight_output * sizeof(float), 1, wfp);
  fread(farr, len_bias_output * sizeof(float), 1, fp);
  fwrite(farr, len_bias_output * sizeof(float), 1, wfp);

  fclose(fp);
  fflush(wfp);
  fclose(wfp);
  return 0;
}

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s INPUT OUTPUT\n", argv[0]);
    return 1;
  }
  if (ConvertModelFile(argv[1], argv[2])) return 1;
  return 0;
}
