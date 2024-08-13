use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...
        // };
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_view = safetensor
                .tensor(name)
                .unwrap_or_else(|_| panic!("TensorNotFound({})", name));
            let data: Vec<f32> = tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            let shape = tensor_view.shape().to_vec();
            Tensor::new(data, &shape)
        };
        let n_layers = config.num_hidden_layers;

        let w_gate = (0..n_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
            .collect();
        let w_down = (0..n_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
            .collect();
        let w_up = (0..n_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
            .collect();
        let lm_head = get_tensor("lm_head.weight");
        let rms_out_w = get_tensor("model.norm.weight");
        let rms_att_w = (0..n_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
            .collect();
        let wq = (0..n_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
            .collect();
        let wk = (0..n_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
            .collect();
        let wv = (0..n_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
            .collect();
        let wo = (0..n_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
            .collect();
        let rms_ffn_w = (0..n_layers)
            .map(|i| {
                get_tensor(&format!(
                    "model.layers.{}.post_attention_layernorm.weight",
                    i
                ))
            })
            .collect();
        let embedding_table = get_tensor("lm_head.weight");

        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}
