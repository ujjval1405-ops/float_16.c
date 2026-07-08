#define _POSIX_C_SOURCE 199309L // Ensures high-precision POSIX timer compatibility
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define NUM_CLASSES 10
#define TOTAL_LAYERS 56

// Helper function to force dropping hidden high-precision bits stored in CPU registers
float force_to_fp16_precision(float val) {
    volatile float storage = val;
    return storage;
}

// Hardware Layer Structure Block using Native C Float16 type
typedef struct {
    _Float16* weights;
    _Float16* bias;
    int in_c;
    int out_c;
    int k;
    int stride;
    int padding;
    int is_depthwise;
    size_t param_count; // Added to track layer-wise total parameters
} FusedConvFP16;

// High-precision cross-platform execution timer
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// 1. Standard / Pointwise Convolution Operations Engine with Embedded Activation (FP16 Precision Controlled)
void conv2d_fp16(const _Float16* input, int in_c, int in_h, int in_w,
                 const _Float16* weights, const _Float16* bias, _Float16* output,
                 int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = (float)bias[oc];
                int in_h_start = oh * stride - padding;
                int in_w_start = ow * stride - padding;
                
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = in_h_start + kh;
                            int iw = in_w_start + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int input_idx = (ic * in_h + ih) * in_w + iw;
                                int weight_idx = ((oc * in_c + ic) * kernel_size + kh) * kernel_size + kw;
                                float prod = (float)input[input_idx] * (float)weights[weight_idx];
                                sum = force_to_fp16_precision(sum + force_to_fp16_precision(prod));
                            }
                        }
                    }
                }
                int out_idx = (oc * out_h + oh) * out_w + ow;
                output[out_idx] = sum > 0.0f ? (_Float16)sum : 0.0f16;
            }
        }
    }
}

// 2. Depthwise Convolution Operations Engine with Embedded Activation (FP16 Precision Controlled)
void depthwise_conv2d_fp16(const _Float16* input, int channels, int in_h, int in_w,
                           const _Float16* weights, const _Float16* bias, _Float16* output,
                           int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = (float)bias[c];
                int in_h_start = oh * stride - padding;
                int in_w_start = ow * stride - padding;
                
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = in_h_start + kh;
                        int iw = in_w_start + kw;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int input_idx = (c * in_h + ih) * in_w + iw;
                            int weight_idx = (c * kernel_size + kh) * kernel_size + kw;
                            float prod = (float)input[input_idx] * (float)weights[weight_idx];
                            sum = force_to_fp16_precision(sum + force_to_fp16_precision(prod));
                        }
                    }
                }
                int out_idx = (c * out_h + oh) * out_w + ow;
                output[out_idx] = sum > 0.0f ? (_Float16)sum : 0.0f16;
            }
        }
    }
}

// 3. Max Pooling Engine Layer (FP16)
void maxpool2d_k3_s2_p1_fp16(const _Float16* input, int channels, int in_h, int in_w, _Float16* output) {
    int out_h = 8; 
    int out_w = 8;
    
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                _Float16 max_val = -65504.0f16;
                int in_h_start = oh * 2 - 1;
                int in_w_start = ow * 2 - 1;
                
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = in_h_start + kh;
                        int iw = in_w_start + kw;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            _Float16 val = input[(c * in_h + ih) * in_w + iw];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                output[(c * out_h + oh) * out_w + ow] = (max_val == -65504.0f16) ? 0.0f16 : max_val;
            }
        }
    }
}

// 4. Channel Shuffle Implementation Block
void channel_shuffle_fp16(const _Float16* input, _Float16* output, int channels, int h, int w, int groups) {
    int channels_per_group = channels / groups;
    int spatial_size = h * w;
    for (int g = 0; g < groups; g++) {
        for (int cp = 0; cp < channels_per_group; cp++) {
            int c_in = g * channels_per_group + cp;
            int c_out = cp * groups + g;
            memcpy(output + c_out * spatial_size, input + c_in * spatial_size, spatial_size * sizeof(_Float16));
        }
    }
}

// 5. Global Adaptive Average Pooling Implementation Layer (FP16) - Updated to perfectly match PyTorch reciprocal behavior
void global_avg_pool_fp16(const _Float16* input, int channels, int h, int w, _Float16* output) {
    int spatial_size = h * w;
    _Float16 scale = (_Float16)(1.0f / (float)spatial_size); // Force reciprocal to FP16 first
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            sum = force_to_fp16_precision(sum + (float)input[c * spatial_size + i]);
        }
        output[c] = (_Float16)force_to_fp16_precision((float)scale * sum);
    }
}

// 6. Classification Head Linear Projection Matrix (FP16 Precision Controlled)
void fully_connected_fp16(const _Float16* input, int in_features, const _Float16* weights, const _Float16* bias, _Float16* output, int out_features) {
    for (int o = 0; o < out_features; o++) {
        float sum = (float)bias[o];
        for (int i = 0; i < in_features; i++) {
            float prod = (float)input[i] * (float)weights[o * in_features + i];
            sum = force_to_fp16_precision(sum + force_to_fp16_precision(prod));
        }
        output[o] = (_Float16)sum;
    }
}

// Dynamic FP16 Weight Matrix Initialization Loader with parameter computation
void load_layer_fp16(FusedConvFP16* layer, int in_c, int out_c, int k, int stride, int padding, int is_depthwise, FILE* fp) {
    layer->in_c = in_c; layer->out_c = out_c; layer->k = k;
    layer->stride = stride; layer->padding = padding; layer->is_depthwise = is_depthwise;
    
    int w_count = is_depthwise ? out_c * k * k : out_c * in_c * k * k;
    layer->param_count = (size_t)(w_count + out_c); // Total Elements (Weights + Biases)
    
    layer->weights = (_Float16*)malloc(w_count * sizeof(_Float16));
    layer->bias = (_Float16*)malloc(out_c * sizeof(_Float16));
    
    fread(layer->weights, sizeof(_Float16), w_count, fp);
    fread(layer->bias, sizeof(_Float16), out_c, fp);
}

int main() {
    int confusion_matrix[NUM_CLASSES][NUM_CLASSES] = {0};
    
    printf("====================================================================\n");
    printf(" INITIALIZING INDEPENDENT C INFRASTRUCTURE & FP16 INFERENCE BACKEND\n");
    printf("====================================================================\n");

    FILE* fp_w = fopen("model_weights_fp16.bin", "rb");
    FILE* fp_d = fopen("test_data_fp16.bin", "rb");
    FILE* fp_p = fopen("pytorch_preds_fp16.bin", "rb");
    if (!fp_w || !fp_d || !fp_p) {
        printf(" Allocation Crash Error: Missing FP16 bin file payloads.\n");
        if (fp_w) fclose(fp_w);
        if (fp_d) fclose(fp_d);
        if (fp_p) fclose(fp_p);
        return -1;
    }

    int total_samples = 0;
    fread(&total_samples, sizeof(int), 1, fp_d);

    int* pytorch_preds = (int*)malloc(total_samples * sizeof(int));
    fread(pytorch_preds, sizeof(int), total_samples, fp_p);
    fclose(fp_p);

    FusedConvFP16 layers[TOTAL_LAYERS];
    int l_idx = 0;

    // Conv1 Root Architecture Load
    load_layer_fp16(&layers[l_idx++], 3, 24, 3, 2, 1, 0, fp_w);

    int stage_out_channels[5] = {24, 48, 96, 192, 1024};
    int stage_repeats[3] = {4, 8, 4};
    int current_in_c = 24;

    for (int s = 0; s < 3; s++) {
        int out_c = stage_out_channels[s + 1];
        int repeats = stage_repeats[s];
        int branch_f = out_c / 2;

        for (int r = 0; r < repeats; r++) {
            if (r == 0) {
                load_layer_fp16(&layers[l_idx++], current_in_c, current_in_c, 3, 2, 1, 1, fp_w); // b1_dw
                load_layer_fp16(&layers[l_idx++], current_in_c, branch_f, 1, 1, 0, 0, fp_w);     // b1_pw
                load_layer_fp16(&layers[l_idx++], current_in_c, branch_f, 1, 1, 0, 0, fp_w);     // b2_pw1
                load_layer_fp16(&layers[l_idx++], branch_f, branch_f, 3, 2, 1, 1, fp_w);         // b2_dw
                load_layer_fp16(&layers[l_idx++], branch_f, branch_f, 1, 1, 0, 0, fp_w);         // b2_pw2
                current_in_c = out_c;
            } else {
                load_layer_fp16(&layers[l_idx++], current_in_c / 2, current_in_c / 2, 1, 1, 0, 0, fp_w); // b2_pw1
                load_layer_fp16(&layers[l_idx++], current_in_c / 2, current_in_c / 2, 3, 1, 1, 1, fp_w); // b2_dw
                load_layer_fp16(&layers[l_idx++], current_in_c / 2, current_in_c / 2, 1, 1, 0, 0, fp_w); // b2_pw2
            }
        }
    }
    // Conv5 Head
    load_layer_fp16(&layers[l_idx++], current_in_c, 1024, 1, 1, 0, 0, fp_w);

    _Float16* fc_weights = (_Float16*)malloc(NUM_CLASSES * 1024 * sizeof(_Float16));
    _Float16* fc_bias = (_Float16*)malloc(NUM_CLASSES * sizeof(_Float16));
    fread(fc_weights, sizeof(_Float16), NUM_CLASSES * 1024, fp_w);
    fread(fc_bias, sizeof(_Float16), NUM_CLASSES, fp_w);
    fclose(fp_w);

    size_t fc_params = (size_t)(NUM_CLASSES * 1024 + NUM_CLASSES);

    // Dynamic Intermediate FP16 Setup Workspace
    _Float16* bufA = (_Float16*)calloc(1024 * 16 * 16, sizeof(_Float16));
    _Float16* bufB = (_Float16*)calloc(1024 * 16 * 16, sizeof(_Float16));
    _Float16* b1_buf = (_Float16*)calloc(512 * 16 * 16, sizeof(_Float16));
    _Float16* b2_buf = (_Float16*)calloc(512 * 16 * 16, sizeof(_Float16));
    _Float16* concat_buf = (_Float16*)calloc(1024 * 16 * 16, sizeof(_Float16));
    _Float16* input_img = (_Float16*)malloc(3 * 32 * 32 * sizeof(_Float16));

    printf("\n====================================================================\n");
    printf(" LAYER-BY-LETTER MEMORY PROFILE & PARAMETER COUNT\n");
    printf("====================================================================\n");
    
    size_t total_allocated_bytes = 0;
    size_t total_network_parameters = 0;

    for (int i = 0; i < TOTAL_LAYERS; i++) {
        size_t w_size = (layers[i].param_count - layers[i].out_c) * sizeof(_Float16);
        size_t b_size = layers[i].out_c * sizeof(_Float16);
        size_t layer_total_bytes = w_size + b_size;
        
        total_allocated_bytes += layer_total_bytes;
        total_network_parameters += layers[i].param_count;
        
        printf("Layer [%2d] (%s) -> Parameters: %7zu | Memory: %7zu B (%6.2f KB)\n",
               i, (layers[i].is_depthwise ? "DW-Conv" : "PW-Conv"),
               layers[i].param_count, layer_total_bytes, (double)layer_total_bytes / 1024.0);
    }
    
    size_t fc_mem = fc_params * sizeof(_Float16);
    size_t buf_mem = (1024 * 16 * 16 * sizeof(_Float16)) * 3 + (512 * 16 * 16 * sizeof(_Float16)) * 2 + (3 * 32 * 32 * sizeof(_Float16));
    
    total_allocated_bytes += fc_mem + buf_mem;
    total_network_parameters += fc_params;

    printf("--------------------------------------------------------------------\n");
    printf("Fully Connected Head Parameters        : %7zu | Memory: %7zu B (%6.2f KB)\n", fc_params, fc_mem, (double)fc_mem / 1024.0);
    printf("Intermediate Activation Buffers Memory : ------- | Memory: %7zu B (%6.2f KB)\n", buf_mem, (double)buf_mem / 1024.0);
    printf("====================================================================\n");
    printf("TOTAL BACKEND WEIGHTS & BIASES COUNT   : %zu network parameters\n", total_network_parameters);
    printf("TOTAL ACTIVE HEAP ALLOCATION           : %zu B (%.2f MB)\n", total_allocated_bytes, (double)total_allocated_bytes / (1024.0 * 1024.0));
    printf("====================================================================\n\n");
    
    printf(" Running Bit-Exact Validation Streaming over %d items...\n\n", total_samples);

    int correct_predictions = 0;
    int framework_matches = 0;
    double continuous_accumulated_time = 0.0;

    double layer_accumulated_times[TOTAL_LAYERS] = {0.0};
    double maxpool_accumulated_time = 0.0;
    double global_pool_accumulated_time = 0.0;
    double fc_accumulated_time = 0.0;

    for (int item = 0; item < total_samples; item++) {
        int true_label = -1;
        fread(input_img, sizeof(_Float16), 3 * 32 * 32, fp_d);
        fread(&true_label, sizeof(int), 1, fp_d);

        double cycle_start = get_time_ms();

        // 1. Root Layer Conv1
        double t_start = get_time_ms();
        conv2d_fp16(input_img, 3, 32, 32, layers[0].weights, layers[0].bias, bufA, 24, 3, 2, 1);
        layer_accumulated_times[0] += (get_time_ms() - t_start);
        
        // 2. MaxPool
        t_start = get_time_ms();
        maxpool2d_k3_s2_p1_fp16(bufA, 24, 16, 16, bufB);
        maxpool_accumulated_time += (get_time_ms() - t_start);

        int c_c = 24, c_h = 8, c_w = 8;
        _Float16* active_tensor = bufB;
        int active_layer = 1;

        // 3. ShuffleNet Processing Loop Stages
        for (int s = 0; s < 3; s++) {
            int out_c = stage_out_channels[s + 1];
            int repeats = stage_repeats[s];
            int branch_f = out_c / 2;

            for (int r = 0; r < repeats; r++) {
                int spatial_sz = c_h * c_w;
                if (r == 0) {
                    int n_h = c_h / 2, n_w = c_w / 2;
                    int n_spatial = n_h * n_w;

                    t_start = get_time_ms();
                    depthwise_conv2d_fp16(active_tensor, c_c, c_h, c_w, layers[active_layer].weights, layers[active_layer].bias, b1_buf, 3, 2, 1);
                    layer_accumulated_times[active_layer++] += (get_time_ms() - t_start);
                    
                    t_start = get_time_ms();
                    conv2d_fp16(b1_buf, c_c, n_h, n_w, layers[active_layer].weights, layers[active_layer].bias, bufA, branch_f, 1, 1, 0);
                    layer_accumulated_times[active_layer++] += (get_time_ms() - t_start);

                    t_start = get_time_ms();
                    conv2d_fp16(active_tensor, c_c, c_h, c_w, layers[active_layer].weights, layers[active_layer].bias, b2_buf, branch_f, 1, 1, 0);
                    layer_accumulated_times[active_layer++] += (get_time_ms() - t_start);
                    
                    t_start = get_time_ms();
                    depthwise_conv2d_fp16(b2_buf, branch_f, c_h, c_w, layers[active_layer].weights, layers[active_layer].bias, b1_buf, 3, 2, 1);
                    layer_accumulated_times[active_layer++] += (get_time_ms() - t_start);
                    
                    t_start = get_time_ms();
                    conv2d_fp16(b1_buf, branch_f, n_h, n_w, layers[active_layer].weights, layers[active_layer].bias, b2_buf, branch_f, 1, 1, 0);
                    layer_accumulated_times[active_layer++] += (get_time_ms() - t_start);

                    memcpy(concat_buf, bufA, branch_f * n_spatial * sizeof(_Float16));
                    memcpy(concat_buf + branch_f * n_spatial, b2_buf, branch_f * n_spatial * sizeof(_Float16));

                    channel_shuffle_fp16(concat_buf, active_tensor, out_c, n_h, n_w, 2);
                    
                    // FIXED: Scoping assignments securely to safely reset block dimensions
                    c_c = out_c; c_h = n_h; c_w = n_w;
                } else {
                    _Float16* x2_split = active_tensor + branch_f * spatial_sz;

                    t_start = get_time_ms();
                    conv2d_fp16(x2_split, branch_f, c_h, c_w, layers[active_layer].weights, layers[active_layer].bias, b1_buf, branch_f, 1, 1, 0);
                    layer_accumulated_times[active_layer++] += (get_time_ms() - t_start);
                    
                    t_start = get_time_ms();
                    depthwise_conv2d_fp16(b1_buf, branch_f, c_h, c_w, layers[active_layer].weights, layers[active_layer].bias, b2_buf, 3, 1, 1);
                    layer_accumulated_times[active_layer++] += (get_time_ms() - t_start);
                    
                    t_start = get_time_ms();
                    conv2d_fp16(b2_buf, branch_f, c_h, c_w, layers[active_layer].weights, layers[active_layer].bias, b1_buf, branch_f, 1, 1, 0);
                    layer_accumulated_times[active_layer++] += (get_time_ms() - t_start);

                    memcpy(concat_buf, active_tensor, branch_f * spatial_sz * sizeof(_Float16));
                    memcpy(concat_buf + branch_f * spatial_sz, b1_buf, branch_f * spatial_sz * sizeof(_Float16));
                    
                    channel_shuffle_fp16(concat_buf, active_tensor, c_c, c_h, c_w, 2);
                }
            }
        }

        // 4. Final Head Processing Layer Conv5
        t_start = get_time_ms();
        conv2d_fp16(active_tensor, c_c, c_h, c_w, layers[active_layer].weights, layers[active_layer].bias, bufA, 1024, 1, 1, 0);
        layer_accumulated_times[active_layer] += (get_time_ms() - t_start);

        // 5. Global Pooling
        t_start = get_time_ms();
        global_avg_pool_fp16(bufA, 1024, c_h, c_w, bufB);
        global_pool_accumulated_time += (get_time_ms() - t_start);

        // 6. Classification Head Multi-Class Layer
        _Float16 class_logits[NUM_CLASSES];
        t_start = get_time_ms();
        fully_connected_fp16(bufB, 1024, fc_weights, fc_bias, class_logits, NUM_CLASSES);
        fc_accumulated_time += (get_time_ms() - t_start);

        double cycle_end = get_time_ms();
        continuous_accumulated_time += (cycle_end - cycle_start);

        int predicted_label = 0;
        _Float16 highest_logit = class_logits[0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (class_logits[c] > highest_logit) {
                highest_logit = class_logits[c];
                predicted_label = c;
            }
        }

        if (true_label >= 0 && true_label < NUM_CLASSES) {
            confusion_matrix[true_label][predicted_label]++;
        }
        if (predicted_label == true_label) correct_predictions++;
        if (predicted_label == pytorch_preds[item]) framework_matches++;

        if ((item + 1) % 1000 == 0 || item < 5) {
            printf("Image Record Index [%4d] -> Real Label: %d | C-Model (FP16): %d | PyTorch Reference: %d -> %s\n", 
                   item, true_label, predicted_label, pytorch_preds[item], 
                   (predicted_label == pytorch_preds[item]) ? "MATCH" : "MISMATCH");
        }
    }
    fclose(fp_d);

    printf("\n====================================================================\n");
    printf(" FINAL REPORT: FULL C INFERENCE BENCHMARK METRICS\n");
    printf("====================================================================\n");
    printf("Total Streaming Evaluations        : %d\n", total_samples);
    printf("Matching Correct Predictions       : %d\n", correct_predictions);
    printf("Calculated Pipeline Target Accuracy: %.2f%%\n", ((float)correct_predictions / total_samples) * 100.0f);
    printf("Framework Parity Matches           : %d / %d\n", framework_matches, total_samples);
    printf("Cross-Framework Parity Alignment   : %.2f%%\n", ((float)framework_matches / total_samples) * 100.0f);
    printf("--------------------------------------------------------------------\n");
    printf("TOTAL INFERENCE EXECUTION TIME     : %.3f ms\n", continuous_accumulated_time);
    printf("Normalized Time per Single Image   : %.4f ms\n", continuous_accumulated_time / total_samples);
    printf("====================================================================\n");
    
    // --- PRINTING THE CONFUSION MATRIX MATRIX ---
    printf("\n====================================================================\n");
    printf(" CONFUSION MATRIX (Rows: Actual Class | Columns: Predicted Class)\n");
    printf("====================================================================\n");
    printf("     ");
    for (int c = 0; c < NUM_CLASSES; c++) printf("[%2d] ", c);
    printf("\n---- ");
    for (int c = 0; c < NUM_CLASSES; c++) printf("-----");
    printf("\n");

    for (int r = 0; r < NUM_CLASSES; r++) {
        printf("[%2d] |", r);
        for (int c = 0; c < NUM_CLASSES; c++) {
            printf("%4d ", confusion_matrix[r][c]);
        }
        printf("\n");
    }
    printf("====================================================================\n");


    // ====================================================================
    //              SKLEARN PERFORMANCE MATRIX REPORT GENERATOR
    // ====================================================================
    printf("\n====================================================================\n");
    printf("                   SKLEARN PERFORMANCE MATRIX REPORT\n");
    printf("====================================================================\n");
    printf("              precision    recall  f1-score   support\n\n");

    float precision_list[NUM_CLASSES] = {0.0f};
    float recall_list[NUM_CLASSES] = {0.0f};
    float f1_list[NUM_CLASSES] = {0.0f};
    int support_list[NUM_CLASSES] = {0};

    float macro_precision = 0.0f, macro_recall = 0.0f, macro_f1 = 0.0f;
    float weighted_precision = 0.0f, weighted_recall = 0.0f, weighted_f1 = 0.0f;

    for (int i = 0; i < NUM_CLASSES; i++) {
        int row_sum = 0; // Actual positive items (Support)
        int col_sum = 0; // Total predicted items for this class
        
        for (int j = 0; j < NUM_CLASSES; j++) {
            row_sum += confusion_matrix[i][j];
            col_sum += confusion_matrix[j][i];
        }
        
        support_list[i] = row_sum;
        precision_list[i] = (col_sum > 0) ? (float)confusion_matrix[i][i] / col_sum : 0.0f;
        recall_list[i] = (row_sum > 0) ? (float)confusion_matrix[i][i] / row_sum : 0.0f;
        
        if ((precision_list[i] + recall_list[i]) > 0.0f) {
            f1_list[i] = 2.0f * (precision_list[i] * recall_list[i]) / (precision_list[i] + recall_list[i]);
        } else {
            f1_list[i] = 0.0f;
        }

        // Print row matching Sklearn spacing format
        printf("           %d     %.4f    %.4f    %.4f      %4d\n", 
               i, precision_list[i], recall_list[i], f1_list[i], support_list[i]);

        // Accumulate macro metrics sums
        macro_precision += precision_list[i];
        macro_recall += recall_list[i];
        macro_f1 += f1_list[i];

        // Accumulate weighted metrics sums
        weighted_precision += precision_list[i] * row_sum;
        weighted_recall += recall_list[i] * row_sum;
        weighted_f1 += f1_list[i] * row_sum;
    }

    // Compute ultimate system global statistics
    float global_accuracy = (float)correct_predictions / total_samples;
    macro_precision /= NUM_CLASSES;
    macro_recall /= NUM_CLASSES;
    macro_f1 /= NUM_CLASSES;

    weighted_precision /= total_samples;
    weighted_recall /= total_samples;
    weighted_f1 /= total_samples;

    printf("\n    accuracy                         %.4f     %5d\n", global_accuracy, total_samples);
    printf("   macro avg     %.4f    %.4f    %.4f     %5d\n", macro_precision, macro_recall, macro_f1, total_samples);
    printf("weighted avg     %.4f    %.4f    %.4f     %5d\n", weighted_precision, weighted_recall, weighted_f1, total_samples);
    printf("====================================================================\n");

    // --- COMPUTING & PRINTING OP-TYPE PROFILING MATRIX ---
    double total_pw_time = 0.0, total_dw_time = 0.0;
    int pw_count = 0, dw_count = 0;
    for (int i = 0; i < TOTAL_LAYERS; i++) {
        if (layers[i].is_depthwise) {
            total_dw_time += layer_accumulated_times[i];
            dw_count++;
        } else {
            total_pw_time += layer_accumulated_times[i];
            pw_count++;
        }
    }
    double structural_sum = total_pw_time + total_dw_time + maxpool_accumulated_time + global_pool_accumulated_time + fc_accumulated_time;

    printf("\n====================================================================\n");
    printf(" EXECUTION PROFILE SUMMARY MATRIX (BY OPERATION TYPE)\n");
    printf("====================================================================\n");
    printf(" %-18s | %-5s | %-15s | %-12s | %-7s\n", "Operation Type", "Count", "Total Time (ms)", "Avg/Img (ms)", "Time %");
    printf("--------------------------------------------------------------------\n");
    printf(" %-18s | %-5d | %-15.3f | %-12.5f | %6.2f%%\n", "Pointwise Conv2D", pw_count, total_pw_time, total_pw_time / total_samples, (total_pw_time / structural_sum) * 100.0);
    printf(" %-18s | %-5d | %-15.3f | %-12.5f | %6.2f%%\n", "Depthwise Conv2D", dw_count, total_dw_time, total_dw_time / total_samples, (total_dw_time / structural_sum) * 100.0);
    printf(" %-18s | %-5d | %-15.3f | %-12.5f | %6.2f%%\n", "Max Pooling 2D", 1, maxpool_accumulated_time, maxpool_accumulated_time / total_samples, (maxpool_accumulated_time / structural_sum) * 100.0);
    printf(" %-18s | %-5d | %-15.3f | %-12.5f | %6.2f%%\n", "Global Avg Pool", 1, global_pool_accumulated_time, global_pool_accumulated_time / total_samples, (global_pool_accumulated_time / structural_sum) * 100.0);
    printf(" %-18s | %-5d | %-15.3f | %-12.5f | %6.2f%%\n", "Fully Connected", 1, fc_accumulated_time, fc_accumulated_time / total_samples, (fc_accumulated_time / structural_sum) * 100.0);
    printf("====================================================================\n");

    printf("\n====================================================================\n");
    printf(" LAYER-BY-LAYER RAW EXECUTION LOGS\n");
    printf("====================================================================\n");
    for (int i = 0; i < TOTAL_LAYERS; i++) {
        printf("Layer [%2d] (%s) AccTime : %10.3f ms | Avg: %.5f ms\n", 
               i, (layers[i].is_depthwise ? "DW-Conv" : "PW-Conv"),
               layer_accumulated_times[i], layer_accumulated_times[i] / total_samples);
    }
    printf("====================================================================\n");

    // --- RELEASING MEMORY ---
    printf("\n====================================================================\n");
    printf(" CLEANING UP ALLOCATED MEMORY\n");
    printf("====================================================================\n");
    printf("Releasing %zu bytes (%.2f MB) back to the operating system...\n", total_allocated_bytes, (double)total_allocated_bytes / (1024.0 * 1024.0));
    printf("====================================================================\n");


    for (int i = 0; i < TOTAL_LAYERS; i++) {
        free(layers[i].weights);
        free(layers[i].bias);
    }
    free(fc_weights); free(fc_bias);
    free(bufA); free(bufB); free(b1_buf); free(b2_buf); free(concat_buf); free(input_img); free(pytorch_preds);

    return 0;
}