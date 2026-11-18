#include "nns/nns.h"
#include "util.h"
#include <cassert>


typedef TransposeLayer::dim Ldims;

static lid_t add_attention(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t decode_len,
		lid_t prev
){

	lid_t Q, K, V, QK, QK_elt, QKV;
	Network::layer_set Ks;
	Q = n.add(NLAYER(name + "_Q", Conv, H=len, W=1, C=numG*gSize), {prev});

	len_t kv_len;
	if(decode_len == 0){
		kv_len = len;
		for(len_t i=0; i<numG; ++i){
			n.add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prev});
			K = n.add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C));
			Ks.push_back(K);
		}
		// K = n.add(NLAYER(name + "_K", PTP, K=numG*len, H=gSize, W=1), Ks);
		V = n.add(NLAYER(name + "_V", Conv, C=numG*gSize, H=len, W=1), {prev});
	}else{
		assert(len == 1);
		kv_len = len + decode_len;

		InputData extK(name + "_Kext", fmap_shape(numG*decode_len, gSize, 1));
		InputData extV(name + "_Vext", fmap_shape(decode_len, numG*gSize, 1));

		for(len_t i=0; i<numG; ++i){
			n.add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prev});
			K = n.add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C));
			Ks.push_back(K);
		}
		K = n.add(NLAYER(name + "_K", PTP, K=numG*kv_len, H=gSize, W=1), Ks, 0, {extK});
		Ks = {K};
		V = n.add(NLAYER(name + "_V", Conv, C=numG*gSize, H=len, W=1), {prev});
		V = n.add(NLAYER(name + "_Vt1", Transpose, K=len, H=numG*gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {V});
		V = n.add(NLAYER(name + "_Vt2", Transpose, K=numG*gSize, H=kv_len, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {V}, 0, {extV});
	}
	QK = n.add(NLAYER(name + "_QK", GroupConv, H=len, W=1, C=numG*gSize, K = numG*kv_len, G=numG), {Q}, 0, {}, Ks);
	QK_elt = n.add(NLAYER(name + "_QK_elt", PTP, K=numG*kv_len, H=len, W=1), {QK});
	QKV = n.add(NLAYER(name + "_QKV", GroupConv, H=len, W=1, C=numG*kv_len, K=numG*gSize, G=numG), {QK_elt}, 0, {}, {V});
	return n.add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), {QKV});
}

static lid_t add_trans_block(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t ff_len, len_t decode_len,
		lid_t prev
){

	lid_t next_prev;
	next_prev = add_attention(n, name, len, numG, gSize, decode_len, prev);
	prev = n.add(NLAYER(name + "_elt1", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	n.add(NLAYER(name + "_feedfwd1", Conv, C=numG*gSize, K=ff_len, H=len, W=1));
	next_prev = n.add(NLAYER(name + "_feedfwd2", Conv, C=ff_len, K=numG*gSize, H=len, W=1));
	return n.add(NLAYER(name + "_elt2", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
}

static Network create_transformer(
		len_t numG, len_t gSize, len_t nBlock, bool is_prefill,
		len_t vocab_len = 1000, len_t len = 512, len_t ff_len = 0
){
	// Default settings.
	if(ff_len == 0){
		ff_len = 4 * len;
	}

	len_t decode_len = 0;
	if(!is_prefill){
		decode_len = len;
		len = 1;
	}

	// Length of embedding
	len_t totG = numG * gSize;
	// Number of embedding
	len_t curH = len;

	lid_t block_prev;
	Network::layer_set prevs;
	Network n;

	InputData input_layer("input_layer", fmap_shape(totG, curH, 1));
	block_prev = n.add(NLAYER("word_embed", PTP, K=totG, H=curH, W=1), {}, 0, {input_layer});
	for(len_t i=1; i<=nBlock; ++i){
		block_prev = add_trans_block(n, "block"+std::to_string(i), len, numG, gSize, ff_len, decode_len, block_prev);
	}
	n.add(NLAYER("proj", Conv, C=totG, K=vocab_len, H=curH, W=1), {block_prev});
	return n;
};

static lid_t add_llama_mha(
		Network& n, const std::string& name,
		len_t len, len_t num_heads, len_t head_size,
		len_t decode_len, lid_t prev
){
	const len_t hidden_size = num_heads * head_size;
	const len_t kv_len = len + decode_len;
	Network::layer_set keys;

	lid_t query = n.add(NLAYER(name + "_Q", Conv,
			C=hidden_size, H=len, W=1), {prev});
	query = n.add(NLAYER(name + "_Q_rope", PTP,
			K=hidden_size, H=len, W=1), {query});

	for(len_t i=0; i<num_heads; ++i){
		const std::string suffix = std::to_string(i);
		lid_t key = n.add(NLAYER(name + "_K" + suffix, Conv,
				C=hidden_size, K=head_size, H=len, W=1), {prev});
		key = n.add(NLAYER(name + "_K_rope" + suffix, PTP,
				K=head_size, H=len, W=1), {key});
		key = n.add(NLAYER(name + "_Kt" + suffix, Transpose,
				K=len, H=head_size, W=1,
				order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {key});
		if(decode_len > 0){
			InputData ext_key(name + "_Kext" + suffix,
					fmap_shape(decode_len, head_size, 1));
			key = n.add(NLAYER(name + "_K_cache" + suffix, PTP,
					K=kv_len, H=head_size, W=1), {key}, 0, {ext_key});
		}
		keys.push_back(key);
	}

	lid_t value = n.add(NLAYER(name + "_V", Conv,
			C=hidden_size, H=len, W=1), {prev});
	if(decode_len > 0){
		value = n.add(NLAYER(name + "_Vt1", Transpose,
				K=len, H=hidden_size, W=1,
				order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {value});
		InputData ext_value(name + "_Vext",
				fmap_shape(decode_len, hidden_size, 1));
		value = n.add(NLAYER(name + "_V_cache", Transpose,
				K=hidden_size, H=kv_len, W=1,
				order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C),
				{value}, 0, {ext_value});
	}

	lid_t scores = n.add(NLAYER(name + "_QK", GroupConv,
			H=len, W=1, C=hidden_size, K=num_heads*kv_len, G=num_heads),
			{query}, 0, {}, keys);
	scores = n.add(NLAYER(name + "_softmax", PTP,
			K=num_heads*kv_len, H=len, W=1), {scores});
	lid_t context = n.add(NLAYER(name + "_QKV", GroupConv,
			H=len, W=1, C=num_heads*kv_len, K=hidden_size, G=num_heads),
			{scores}, 0, {}, {value});
	return n.add(NLAYER(name + "_out_proj", Conv,
			C=hidden_size, H=len, W=1), {context});
}

/*
 * The common MHA case uses the same compact grouped-convolution form as the
 * existing Transformer input.  For GQA, attention is expanded one query head
 * at a time so several query heads can reuse one K/V projection without
 * charging for duplicated K/V projections.
 */
static lid_t add_llama_attention(
		Network& n, const std::string& name,
		len_t len, len_t num_heads, len_t num_kv_heads,
		len_t head_size, len_t decode_len, lid_t prev
){
	assert(num_heads > 0 && num_kv_heads > 0 && head_size > 0);
	assert(num_heads % num_kv_heads == 0);
	if(num_heads == num_kv_heads){
		return add_llama_mha(n, name, len, num_heads, head_size,
				decode_len, prev);
	}

	const len_t hidden_size = num_heads * head_size;
	const len_t kv_len = len + decode_len;
	const len_t queries_per_kv = num_heads / num_kv_heads;
	std::vector<lid_t> keys;
	std::vector<lid_t> values;
	Network::layer_set attention_heads;

	for(len_t i=0; i<num_kv_heads; ++i){
		const std::string suffix = std::to_string(i);
		lid_t key = n.add(NLAYER(name + "_K" + suffix, Conv,
				C=hidden_size, K=head_size, H=len, W=1), {prev});
		key = n.add(NLAYER(name + "_K_rope" + suffix, PTP,
				K=head_size, H=len, W=1), {key});
		key = n.add(NLAYER(name + "_Kt" + suffix, Transpose,
				K=len, H=head_size, W=1,
				order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {key});

		lid_t value = n.add(NLAYER(name + "_V" + suffix, Conv,
				C=hidden_size, K=head_size, H=len, W=1), {prev});

		if(decode_len > 0){
			InputData ext_key(name + "_Kext" + suffix,
					fmap_shape(decode_len, head_size, 1));
			key = n.add(NLAYER(name + "_K_cache" + suffix, PTP,
					K=kv_len, H=head_size, W=1), {key}, 0, {ext_key});

			value = n.add(NLAYER(name + "_Vt1_" + suffix, Transpose,
					K=len, H=head_size, W=1,
					order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {value});
			InputData ext_value(name + "_Vext" + suffix,
					fmap_shape(decode_len, head_size, 1));
			value = n.add(NLAYER(name + "_V_cache" + suffix, Transpose,
					K=head_size, H=kv_len, W=1,
					order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C),
					{value}, 0, {ext_value});
		}

		keys.push_back(key);
		values.push_back(value);
	}

	for(len_t i=0; i<num_heads; ++i){
		const std::string suffix = std::to_string(i);
		const len_t kv_head = i / queries_per_kv;
		lid_t query = n.add(NLAYER(name + "_Q" + suffix, Conv,
				C=hidden_size, K=head_size, H=len, W=1), {prev});
		query = n.add(NLAYER(name + "_Q_rope" + suffix, PTP,
				K=head_size, H=len, W=1), {query});

		lid_t scores = n.add(NLAYER(name + "_QK" + suffix, Conv,
				C=head_size, K=kv_len, H=len, W=1),
				{query}, 0, {}, {keys[kv_head]});
		scores = n.add(NLAYER(name + "_softmax" + suffix, PTP,
				K=kv_len, H=len, W=1), {scores});
		lid_t head = n.add(NLAYER(name + "_QKV" + suffix, Conv,
				C=kv_len, K=head_size, H=len, W=1),
				{scores}, 0, {}, {values[kv_head]});
		attention_heads.push_back(head);
	}

	return n.add(NLAYER(name + "_out_proj", Conv,
			C=hidden_size, K=hidden_size, H=len, W=1), attention_heads);
}

static lid_t add_llama_block(
		Network& n, const std::string& name,
		len_t len, len_t num_heads, len_t num_kv_heads,
		len_t head_size, len_t ff_len, len_t decode_len, lid_t prev
){
	const len_t hidden_size = num_heads * head_size;
	lid_t normalized = n.add(NLAYER(name + "_attn_norm", PTP,
			K=hidden_size, H=len, W=1), {prev});
	lid_t attention = add_llama_attention(n, name + "_attn", len,
			num_heads, num_kv_heads, head_size, decode_len, normalized);
	prev = n.add(NLAYER(name + "_attn_residual", Eltwise,
			K=hidden_size, H=len, W=1, N=2), {prev, attention});

	normalized = n.add(NLAYER(name + "_ffn_norm", PTP,
			K=hidden_size, H=len, W=1), {prev});
	lid_t gate = n.add(NLAYER(name + "_ffn_gate", Conv,
			C=hidden_size, K=ff_len, H=len, W=1), {normalized});
	gate = n.add(NLAYER(name + "_ffn_silu", PTP,
			K=ff_len, H=len, W=1), {gate});
	lid_t up = n.add(NLAYER(name + "_ffn_up", Conv,
			C=hidden_size, K=ff_len, H=len, W=1), {normalized});
	lid_t gated = n.add(NLAYER(name + "_ffn_mul", Eltwise,
			K=ff_len, H=len, W=1, N=2), {gate, up});
	lid_t down = n.add(NLAYER(name + "_ffn_down", Conv,
			C=ff_len, K=hidden_size, H=len, W=1), {gated});
	return n.add(NLAYER(name + "_ffn_residual", Eltwise,
			K=hidden_size, H=len, W=1, N=2), {prev, down});
}

static Network create_llama(
		len_t num_heads, len_t num_kv_heads, len_t head_size,
		len_t ff_len, len_t n_blocks, bool is_prefill,
		len_t vocab_len = 32000, len_t context_len = 512
){
	assert(num_heads > 0 && num_kv_heads > 0 && head_size > 0);
	assert(num_heads % num_kv_heads == 0);
	assert(ff_len > 0 && n_blocks > 0 && context_len > 0);

	len_t len = context_len;
	len_t decode_len = 0;
	if(!is_prefill){
		decode_len = context_len;
		len = 1;
	}

	const len_t hidden_size = num_heads * head_size;
	Network n;
	InputData input_layer("input_layer", fmap_shape(hidden_size, len, 1));
	lid_t block_prev = n.add(NLAYER("word_embed", PTP,
			K=hidden_size, H=len, W=1), {}, 0, {input_layer});
	for(len_t i=1; i<=n_blocks; ++i){
		block_prev = add_llama_block(n, "block" + std::to_string(i), len,
				num_heads, num_kv_heads, head_size, ff_len, decode_len, block_prev);
	}
	block_prev = n.add(NLAYER("output_norm", PTP,
			K=hidden_size, H=len, W=1), {block_prev});
	n.add(NLAYER("proj", Conv,
			C=hidden_size, K=vocab_len, H=len, W=1), {block_prev});
	return n;
}



/*
 * Since the complete networks have too many layers,
 * and all blocks in the network are identical,
 * a single block is provided for each network.
 *
 * To run the full network, one can uncomment the networks below.
 * Also remember to uncomment in "nns/nns.h", add the network in "main.cpp",
 * and increase MAX_BITS_IN_BS in "bitset.h".
 */

/*
 * BERT-Large
 *
 * numG       = 16
 * gSize      = 64
 * nBlock     = 24
 * is_prefill = true
*/
// const Network BERT = create_transformer(16, 64, 24, true);
const Network BERT_block = create_transformer(16, 64, 1, true);

/*
 * GPT2-XL at Prefill stage
 *
 * numG       = 25
 * gSize      = 64
 * nBlock     = 48
 * is_prefill = true
*/
// const Network GPT2_prefill = create_transformer(25, 64, 48, true);
const Network GPT2_prefill_block = create_transformer(25, 64, 1, true);

/*
 * GPT2-XL at Decode stage
 *
 * numG       = 25
 * gSize      = 64
 * nBlock     = 48
 * is_prefill = false
*/
// const Network GPT2_decode = create_transformer(25, 64, 48, false);
const Network GPT2_decode_block = create_transformer(25, 64, 1, false);

/*
 * Llama 2 7B
 *
 * num_heads    = 32
 * num_kv_heads = 32
 * head_size    = 128
 * ff_len       = 11008
 * n_blocks     = 32
 * vocab_len    = 32000
 *
 * As with the other large models above, expose one representative block to
 * keep the scheduling search tractable.  create_llama also accepts fewer KV
 * heads for GQA-based Llama variants.
 */
// LLaMA 2 7B
const Network LLaMA7B_prefill_block =
    create_llama(32, 32, 128, 11008, 1, true);

const Network LLaMA7B_decode_block =
    create_llama(32, 32, 128, 11008, 1, false);


// LLaMA 2 13B
const Network LLaMA13B_prefill_block =
    create_llama(40, 40, 128, 13824, 1, true);

const Network LLaMA13B_decode_block =
    create_llama(40, 40, 128, 13824, 1, false);


// LLaMA 2 34B
const Network LLaMA34B_prefill_block =
    create_llama(64, 8, 128, 22016, 1, true);

const Network LLaMA34B_decode_block =
    create_llama(64, 8, 128, 22016, 1, false);

const Network QWEN25_prefill_block =
    create_llama(
        28,       // num_heads
        4,        // num_kv_heads (GQA)
        128,      // head_size
        18944,    // ff_len
        1,        // n_blocks: one representative block
        true,     // is_prefill
        152064,   // vocab_len
        512       // context_len
    );

const Network QWEN25_decode_block =
    create_llama(
        28,       // num_heads
        4,        // num_kv_heads (GQA)
        128,      // head_size
        18944,    // ff_len
        1,        // n_blocks: one representative block
        false,    // is_prefill
        152064,   // vocab_len
        512       // context_len / cached context length
    );