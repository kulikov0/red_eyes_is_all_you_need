// pos_emb byte ROM plus a combined LN BRAM with 18 packed 128-byte tensors.
// tensor_sel[5:0] + addr -> 8-bit weight data + 16-bit fp16 scale

module weight_store (
  input  wire        clk_i,
  input  wire [ 5:0] tensor_sel_i,  // 0..35
  input  wire [14:0] addr_i,
  output reg  [ 7:0] data_o,
  output reg  [15:0] scale_o
);

  // FP16 scale factors
  `include "weight_scales.vh"

  reg [14:0] addr_r;
  reg [5:0]  sel_r1;
  always @(posedge clk_i) begin
    addr_r <= addr_i;
    sel_r1 <= tensor_sel_i;
  end

  // BRAM ROM outputs (synchronous, 1-cycle latency)
  wire [7:0] d_pos_emb;

  // Combined LayerNorm ROM: 18 tensors x 128 bytes = 2304 in one BRAM
  // Offset LUT maps tensor_sel to base address within ln_params.hex
  wire [7:0] d_ln;
  reg [11:0] ln_offset;
  wire [11:0] ln_addr = ln_offset + {5'd0, addr_r[6:0]};

  // Large weight ROMs

  weight_rom #(
    .DEPTH(32768),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/pos_emb_weight.hex")
  ) u_pos_emb (
    .clk_i (clk_i),
    .addr_i(addr_r),
    .data_o(d_pos_emb)
  );

  //  Combined LayerNorm ROM
  weight_rom #(
    .DEPTH(2304),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/ln_params.hex")
  ) u_ln (
    .clk_i (clk_i),
    .addr_i(ln_addr),
    .data_o(d_ln)
  );

  // Offset LUT: tensor_sel -> base address in ln_params.hex
  // Order matches extract_weights.py: all size-128 tensors in bin order
  // idx 0: b0_ln1_w, 1: b0_ln1_b, 2: b0_ln2_w, ... 17: ln_f_b
  always @(*) begin
    case (sel_r1)
      6'd2:  ln_offset = 12'd0;     // block0_ln1_weight
      6'd3:  ln_offset = 12'd128;   // block0_ln1_bias
      6'd6:  ln_offset = 12'd256;   // block0_ln2_weight
      6'd7:  ln_offset = 12'd384;   // block0_ln2_bias
      6'd10: ln_offset = 12'd512;   // block1_ln1_weight
      6'd11: ln_offset = 12'd640;   // block1_ln1_bias
      6'd14: ln_offset = 12'd768;   // block1_ln2_weight
      6'd15: ln_offset = 12'd896;   // block1_ln2_bias
      6'd18: ln_offset = 12'd1024;  // block2_ln1_weight
      6'd19: ln_offset = 12'd1152;  // block2_ln1_bias
      6'd22: ln_offset = 12'd1280;  // block2_ln2_weight
      6'd23: ln_offset = 12'd1408;  // block2_ln2_bias
      6'd26: ln_offset = 12'd1536;  // block3_ln1_weight
      6'd27: ln_offset = 12'd1664;  // block3_ln1_bias
      6'd30: ln_offset = 12'd1792;  // block3_ln2_weight
      6'd31: ln_offset = 12'd1920;  // block3_ln2_bias
      6'd34: ln_offset = 12'd2048;  // ln_f_weight
      6'd35: ln_offset = 12'd2176;  // ln_f_bias
      default: ln_offset = 12'd0;
    endcase
  end

  // Output MUX: tensor_sel -> data
  // Registered to match BRAM 1-cycle latency
  reg [5:0] sel_r;
  always @(posedge clk_i) sel_r <= sel_r1;

  always @(*) begin
    case (sel_r)
      6'd1:  data_o = d_pos_emb;
      6'd2,
      6'd3,
      6'd6,
      6'd7,
      6'd10,
      6'd11,
      6'd14,
      6'd15,
      6'd18,
      6'd19,
      6'd22,
      6'd23,
      6'd26,
      6'd27,
      6'd30,
      6'd31,
      6'd34,
      6'd35: data_o = d_ln;
      default: data_o = 8'd0;
    endcase
  end

  // Scale LUT: tensor_sel -> fp16 scale
  always @(posedge clk_i) begin
    case (sel_r1)
      6'd1:  scale_o <= SCALE_POS_EMB_WEIGHT;
      6'd2:  scale_o <= SCALE_BLOCK0_LN1_WEIGHT;
      6'd3:  scale_o <= SCALE_BLOCK0_LN1_BIAS;
      6'd4:  scale_o <= SCALE_BLOCK0_ATTN_QKV_WEIGHT;
      6'd5:  scale_o <= SCALE_BLOCK0_ATTN_PROJ_WEIGHT;
      6'd6:  scale_o <= SCALE_BLOCK0_LN2_WEIGHT;
      6'd7:  scale_o <= SCALE_BLOCK0_LN2_BIAS;
      6'd8:  scale_o <= SCALE_BLOCK0_FF_UP_WEIGHT;
      6'd9:  scale_o <= SCALE_BLOCK0_FF_DOWN_WEIGHT;
      6'd10: scale_o <= SCALE_BLOCK1_LN1_WEIGHT;
      6'd11: scale_o <= SCALE_BLOCK1_LN1_BIAS;
      6'd12: scale_o <= SCALE_BLOCK1_ATTN_QKV_WEIGHT;
      6'd13: scale_o <= SCALE_BLOCK1_ATTN_PROJ_WEIGHT;
      6'd14: scale_o <= SCALE_BLOCK1_LN2_WEIGHT;
      6'd15: scale_o <= SCALE_BLOCK1_LN2_BIAS;
      6'd16: scale_o <= SCALE_BLOCK1_FF_UP_WEIGHT;
      6'd17: scale_o <= SCALE_BLOCK1_FF_DOWN_WEIGHT;
      6'd18: scale_o <= SCALE_BLOCK2_LN1_WEIGHT;
      6'd19: scale_o <= SCALE_BLOCK2_LN1_BIAS;
      6'd20: scale_o <= SCALE_BLOCK2_ATTN_QKV_WEIGHT;
      6'd21: scale_o <= SCALE_BLOCK2_ATTN_PROJ_WEIGHT;
      6'd22: scale_o <= SCALE_BLOCK2_LN2_WEIGHT;
      6'd23: scale_o <= SCALE_BLOCK2_LN2_BIAS;
      6'd24: scale_o <= SCALE_BLOCK2_FF_UP_WEIGHT;
      6'd25: scale_o <= SCALE_BLOCK2_FF_DOWN_WEIGHT;
      6'd26: scale_o <= SCALE_BLOCK3_LN1_WEIGHT;
      6'd27: scale_o <= SCALE_BLOCK3_LN1_BIAS;
      6'd28: scale_o <= SCALE_BLOCK3_ATTN_QKV_WEIGHT;
      6'd29: scale_o <= SCALE_BLOCK3_ATTN_PROJ_WEIGHT;
      6'd30: scale_o <= SCALE_BLOCK3_LN2_WEIGHT;
      6'd31: scale_o <= SCALE_BLOCK3_LN2_BIAS;
      6'd32: scale_o <= SCALE_BLOCK3_FF_UP_WEIGHT;
      6'd33: scale_o <= SCALE_BLOCK3_FF_DOWN_WEIGHT;
      6'd34: scale_o <= SCALE_LN_F_WEIGHT;
      6'd35: scale_o <= SCALE_LN_F_BIAS;
      default: scale_o <= 16'd0;
    endcase
  end

endmodule