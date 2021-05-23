#include <stdio.h>
#include <stdlib.h>

const int kWidth = 32, kHeight = 32;
const int kXPadding = 2, kYPadding = 2;
const float kScaleMin = -1, kScaleMax = 1;

unsigned char *labels;
float *images;
int count;

int LogError(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  return -1;
}

int ReverseInt(int x) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = x & 255;
  ch2 = (x >> 8) & 255;
  ch3 = (x >> 16) & 255;
  ch4 = (x >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

int Read(const char *in_name) {
  // open file
  char labels_file[256], images_file[256];
  sprintf(labels_file, "%s-labels-idx1-ubyte", in_name);
  sprintf(images_file, "%s-images-idx3-ubyte", in_name);
  FILE *lfp = fopen(labels_file, "rb");
  FILE *ifp = fopen(images_file, "rb");
  if (!lfp || !ifp) return LogError("failed to open file");
  // read file header
  int temp, llen, ilen, iw, ih;
  fread(&temp, sizeof(int), 1, lfp);
  fread(&llen, sizeof(int), 1, lfp);
  if (ReverseInt(temp) != 0x00000801) return LogError("invalid label file");
  llen = ReverseInt(llen);
  fread(&temp, sizeof(int), 1, ifp);
  fread(&ilen, sizeof(int), 1, ifp);
  fread(&iw, sizeof(int), 1, ifp);
  fread(&ih, sizeof(int), 1, ifp);
  if (ReverseInt(temp) != 0x00000803) return LogError("invalid image file");
  ilen = ReverseInt(ilen);
  if (llen != ilen) return LogError("length mismatch");
  count = llen;
  iw = ReverseInt(iw);
  ih = ReverseInt(ih);
  // read labels
  labels = (unsigned char *)malloc(llen * sizeof(unsigned char));
  for (int i = 0; i < llen; ++i) {
    unsigned char c;
    fread(&c, sizeof(unsigned char), 1, lfp);
    labels[i] = c;
  }
  // read images
  images = (float *)malloc(ilen * kWidth * kHeight * sizeof(float));
  for (int i = 0; i < ilen * kWidth * kHeight; ++i) images[i] = -1;
  for (int i = 0; i < ilen; ++i) {
    int addr = kWidth * kHeight * i;
    for (int r = 0; r < iw; ++r) {
      for (int c = 0; c < ih; ++c) {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, ifp);
        images[addr + kWidth * (r + kYPadding) + c + kXPadding] =
            (temp / 255.0) * (kScaleMax - kScaleMin) + kScaleMin;
      }
    }
  }
  return 0;
}

int DumpImage(const char *out_prefix, int label, int index) {
  // open file
  char file_name[256];
  sprintf(file_name, "%s-%i-%i.bin", out_prefix, index, label);
  FILE *fp = fopen(file_name, "wb");
  if (!fp) return LogError("failed to create file");
  // write data
  float *arr = images + index * kWidth * kHeight;
  fwrite(arr, kWidth * kHeight * sizeof(float), 1, fp);
  return 0;
}

int DumpMnist(const char *out_prefix) {
  for (int i = 0; i < count; ++i) {
    int ret = DumpImage(out_prefix, labels[i], i);
    if (ret) return ret;
  }
  return 0;
}

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s IN_NAME OUT_PREFIX\n", argv[0]);
    return 1;
  }
  if (Read(argv[1]) || DumpMnist(argv[2])) return 1;
  return 0;
}
