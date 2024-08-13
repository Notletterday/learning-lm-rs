use std::ffi::c_short;

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let x_shape = x.shape();
    let y_shape = y.shape();
    let w_shape = w.shape();
    assert!(x_shape == y_shape, "x and y must match");
    assert!(x_shape.len() == 2, "x must be first dimensional");
    assert!(w_shape.len() == 1, "w must be second dimensional");
    assert!(
        x_shape[1] == w_shape[0],
        "w must have equal number of elements as the second dimensional of x"
    );
    let x_data = x.data();
    let w_data = w.data();
    let mut y_data = unsafe { y.data_mut() };
    let n = x_shape[1] as f32;
    for i in 0..x_shape[0] {
        let offset = i * x_shape[1];
        let mut rms = 0.0;
        for j in 0..x_shape[1] {
            rms += x_data[offset + j] * x_data[offset + j];
        }
        rms = (rms / n).sqrt() + epsilon;
        for j in 0..x_shape[1] {
            y_data[offset + j] = x_data[offset + j] / rms * w_data[j];
        }
    }
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    for i in 0..len {
        let sigmoid_x = 1.0 / (1.0 + (-_x[i]).exp());
        _y[i] = sigmoid_x * _x[i] * _y[i];
    }
}
//新加的silu
pub fn silu_two_dimension(z: &mut Tensor<f32>, y: &Tensor<f32>, x: &Tensor<f32>) {
    let y_shape = y.shape();
    let x_shape = x.shape();
    let z_shape = z.shape();
    assert!(y_shape.len() == 2);
    assert!(x_shape.len() == 2);
    assert!(z_shape[0] == x_shape[0] && z_shape[0] == y_shape[0]);
    assert!(z_shape[1] == x_shape[1] && z_shape[1] == y_shape[1]);
    assert!(z_shape.len() == 2);
    let _y = y.data();
    let _x = x.data();
    let _z = unsafe { z.data_mut() };
    for i in 0..x_shape[0] {
        for j in 0..x_shape[1] {
            let t = i * x_shape[1] + j;
            _z[t] = _y[t] / (1.0 + (-_y[t]).exp()) * _x[t];
        }
    }
}
//新加的add
pub fn tensor_add(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let x_shape = x.shape();
    let y_shape = y.shape();
    assert!(x_shape.len() == y_shape.len());
    assert!(x_shape[0] == y_shape[0] && x_shape[1] == y_shape[1]);
    let _x = x.data();
    let _y = unsafe { y.data_mut() };
    for i in 0..x_shape[0] {
        for j in 0..x_shape[1] {
            let t = i * x_shape[1] + j;
            _y[t] += _x[t];
        }
    }
}
// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();
    assert!(a_shape.len() == b_shape.len());
    assert!(a_shape.len() == b_shape.len());
    assert!(a_shape[1] == b_shape[1]);
    assert!(a_shape[0] == c_shape[0]);
    assert!(b_shape[0] == c_shape[1]);
    let c_shape1 = c_shape[1];
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };
    for elem in c_data.iter_mut() {
        *elem *= beta;
    }
    for i in 0..a_shape[0] {
        for j in 0..b_shape[0] {
            let mut sum = 0.0;
            for k in 0..a_shape[1] {
                sum += a_data[i * a_shape[1] + k] * b_data[j * b_shape[1] + k];
            }
            c_data[i * c_shape1 + j] += alpha * sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
