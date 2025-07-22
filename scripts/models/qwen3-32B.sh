MODEL_ARGS=(
   --swiglu
   --num-layers 64
   --hidden-size 5120
   --ffn-hidden-size 25600
   --num-attention-heads 64
   --group-query-attention
   --num-query-groups 8
   --kv-channels 128
   --use-rotary-position-embeddings
   --rotary-base 1000000
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --qk-layernorm
   --untie-embeddings-and-output-weights
   --vocab-size 151936
)