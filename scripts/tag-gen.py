#almost entirely stollen from Kohaku's original hf space - https://huggingface.co/spaces/KBlueLeaf/DTG-demo
import modules.scripts as scripts
import gradio as gr
import os
from contextlib import nullcontext
from random import shuffle
from time import time_ns

from modules import script_callbacks
import torch
from huggingface_hub import Repository
from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

MODEL_PATH = "KBlueLeaf/DanTagGen-beta"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#moved from other file
SPECIAL = [
    "1girl",
    "2girls",
    "3girls",
    "4girls",
    "5girls",
    "6+girls",
    "multiple_girls",
    "1boy",
    "2boys",
    "3boys",
    "4boys",
    "5boys",
    "6+boys",
    "multiple_boys",
    "male_focus",
    "1other",
    "2others",
    "3others",
    "4others",
    "5others",
    "6+others",
    "multiple_others",
]
TARGET = {
    "very_short": 10,
    "short": 20,
    "long": 40,
    "very_long": 60,
}

def generate(
    model: PreTrainedModel | Llama,
    tokenizer: PreTrainedTokenizerBase,
    prompt="",
    temperature=0.5,
    top_p=0.95,
    top_k=45,
    repetition_penalty=1.17,
    max_new_tokens=128,
    autocast_gen=lambda: torch.autocast("cpu", enabled=False),
    **kwargs,
):
    if isinstance(model, Llama):
        result = model.create_completion(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repeat_penalty=repetition_penalty or 1,
        )
        return prompt + result["choices"][0]["text"]

    torch.cuda.empty_cache()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        **kwargs,
    )
    with torch.no_grad(), autocast_gen():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    torch.cuda.empty_cache()
    return output


def tag_gen(
    text_model,
    tokenizer,
    prompt,
    prompt_tags,
    len_target,
    black_list,
    temperature=0.5,
    top_p=0.95,
    top_k=100,
    max_new_tokens=256,
    max_retry=5,
):
    prev_len = 0
    retry = max_retry
    llm_gen = ""

    while True:
        llm_gen = generate(
            model=text_model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens,
            stream_output=False,
            autocast_gen=nullcontext,
            prompt_lookup_num_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        llm_gen = llm_gen.replace("</s>", "").replace("<s>", "")
        extra = llm_gen.split("<|input_end|>")[-1].strip().strip(",")
        extra_tokens = list(
            set(
                [
                    tok.strip()
                    for tok in extra.split(",")
                    if tok.strip() not in black_list
                ]
            )
        )
        llm_gen = llm_gen.replace(extra, ", ".join(extra_tokens))

        yield llm_gen, extra_tokens

        if len(prompt_tags) + len(extra_tokens) < len_target:
            if len(extra_tokens) == prev_len and prev_len > 0:
                if retry < 0:
                    break
                retry -= 1
            shuffle(extra_tokens)
            retry = max_retry
            prev_len = len(extra_tokens)
            prompt = llm_gen.strip().replace("  <|", " <|")
        else:
            break
    yield llm_gen, extra_tokens

def get_result(
    text_model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    rating: str = "",
    artist: str = "",
    characters: str = "",
    copyrights: str = "",
    target: str = "long",
    special_tags: list[str] = ["1girl"],
    general: str = "",
    aspect_ratio: float = 0.0,
    blacklist: str = "",
    escape_bracket: bool = False,
    temperature: float = 1.35,
):
    start = time_ns()
    print("=" * 50, "\n")
    # Use LLM to predict possible summary
    # This prompt allow model itself to make request longer based on what it learned
    # Which will be better for preference sim and pref-sum contrastive scorer
    prompt = f"""
rating: {rating or '<|empty|>'}
artist: {artist.strip() or '<|empty|>'}
characters: {characters.strip() or '<|empty|>'}
copyrights: {copyrights.strip() or '<|empty|>'}
aspect ratio: {f"{aspect_ratio:.1f}" or '<|empty|>'}
target: {'<|' + target + '|>' if target else '<|long|>'}
general: {", ".join(special_tags)}, {general.strip().strip(",")}<|input_end|>
""".strip()

    artist = artist.strip().strip(",").replace("_", " ")
    characters = characters.strip().strip(",").replace("_", " ")
    copyrights = copyrights.strip().strip(",").replace("_", " ")
    special_tags = [tag.strip().replace("_", " ") for tag in special_tags]
    general = general.strip().strip(",")
    black_list = set(
        [tag.strip().replace("_", " ") for tag in blacklist.strip().split(",")]
    )

    prompt_tags = special_tags + general.strip().strip(",").split(",")
    len_target = TARGET[target]
    llm_gen = ""

    for llm_gen, extra_tokens in tag_gen(
        text_model,
        tokenizer,
        prompt,
        prompt_tags,
        len_target,
        black_list,
        temperature=temperature,
        top_p=0.95,
        top_k=100,
        max_new_tokens=256,
        max_retry=5,
    ):
        yield "", llm_gen, f"Total cost time: {(time_ns()-start)/1e9:.2f}s"
    print()
    print("-" * 50)

    general = f"{general.strip().strip(',')}, {','.join(extra_tokens)}"
    tags = general.strip().split(",")
    tags = [tag.strip() for tag in tags if tag.strip()]
    special = special_tags + [tag for tag in tags if tag in SPECIAL]
    tags = [tag for tag in tags if tag not in special]

    final_prompt = ", ".join(special)
    if characters:
        final_prompt += f", \n\n{characters}"
    if copyrights:
        final_prompt += ", "
        if not characters:
            final_prompt += "\n\n"
        final_prompt += copyrights
    if artist:
        final_prompt += f", \n\n{artist}"
    final_prompt += f""", \n\n{', '.join(tags)},

masterpiece, newest, absurdres, {rating}"""

    print(final_prompt)
    print("=" * 50)

    if escape_bracket:
        final_prompt = (
            final_prompt.replace("[", "\\[")
            .replace("]", "\\]")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    yield final_prompt, llm_gen, f"Total cost time: {(time_ns()-start)/1e9:.2f}s  |  Total general tags: {len(special+tags)}"

def wrapper(
    rating: str,
    artist: str,
    characters: str,
    copyrights: str,
    target: str,
    special_tags: list[str],
    general: str,
    width: float,
    height: float,
    blacklist: str,
    escape_bracket: bool,
    temperature: float = 1.35,
):
    print("DTG: Loading text model")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    text_model = LlamaForCausalLM.from_pretrained(MODEL_PATH)
    text_model = text_model.eval().half().to(DEVICE)
    yield from get_result(
        text_model,
        tokenizer,
        rating,
        artist,
        characters,
        copyrights,
        target,
        special_tags,
        general,
        width / height,
        blacklist,
        escape_bracket,
        temperature,
    )
    #unload after use
    del tokenizer
    del text_model
    print("DTG: Unloaded text model")

#send to txt2img
send = '''
let textarea = gradioApp().getElementById(id).querySelector('textarea');
textarea.value = ("test");
'''
def notify():
    print("Sent prompt to txt2img")

#UI
def on_ui_tabs():
    with gr.Blocks(theme=gr.themes.Soft(),analytics_enabled=False) as ui_component:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column(scale=2):
                        rating = gr.Radio(
                            ["safe", "sensitive", "nsfw", "nsfw, explicit"],
                            value="safe",
                            label="Rating",
                        )
                        special_tags = gr.Dropdown(
                            SPECIAL,
                            value=["1girl"],
                            label="Special tags",
                            multiselect=True,
                        )
                        characters = gr.Textbox(label="Characters")
                        copyrights = gr.Textbox(label="Copyrights(Series)")
                        artist = gr.Textbox(label="Artist")
                        target = gr.Radio(
                            ["very_short", "short", "long", "very_long"],
                            value="long",
                            label="Target length",
                        )
                    with gr.Column(scale=2):
                        general = gr.TextArea(label="Input your general tags")
                        black_list = gr.TextArea(
                            label="tag Black list (seperated by comma)"
                        )
                        with gr.Row():
                            width = gr.Slider(
                                value=1024,
                                minimum=256,
                                maximum=4096,
                                step=32,
                                label="Width",
                            )
                            height = gr.Slider(
                                value=1024,
                                minimum=256,
                                maximum=4096,
                                step=32,
                                label="Height",
                            )
                        with gr.Row():
                            temperature = gr.Slider(
                                value=1.35,
                                minimum=0.1,
                                maximum=2,
                                step=0.05,
                                label="Temperature",
                            )
                            escape_bracket = gr.Checkbox(
                                value=False,
                                label="Escape bracket",
                            )
                submit = gr.Button("Submit")
            with gr.Column(scale=3):
                formated_result = gr.TextArea(
                    label="Final output", lines=14, show_copy_button=True
                )
                llm_result = gr.TextArea(label="LLM output", lines=10)
                cost_time = gr.Markdown()
        submit.click(
            wrapper,
            inputs=[
                rating,
                artist,
                characters,
                copyrights,
                target,
                special_tags,
                general,
                width,
                height,
                black_list,
                escape_bracket,
                temperature,
            ],
            outputs=[
                formated_result,
                llm_result,
                cost_time,
            ],
            show_progress=True,
        )
    
    return [(ui_component, "Tag-gen", "Tag_Gen_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)