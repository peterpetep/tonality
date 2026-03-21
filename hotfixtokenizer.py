def hotfixtokenizer(tokenizer):
    tokenizer._tokenizer.post_processor  # inspect current post processor

    tokenizer._add_bos_token = True
    tokenizer._add_eos_token = True
    tokenizer.update_post_processor()
