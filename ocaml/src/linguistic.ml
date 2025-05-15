open Core
open Core_kernel
open Angstrom

(* Type definitions *)
type token = {
  text: string;
  pos: string;
  lemma: string;
  features: (string * string) list;
} [@@deriving yojson]

type sentence = token list [@@deriving yojson]

(* Linguistic processing functions *)
let tokenize text =
  let words = String.split ~on:' ' text in
  List.map ~f:(fun w -> {
    text = w;
    pos = "UNK";  (* Placeholder for POS tagging *)
    lemma = w;    (* Placeholder for lemmatization *)
    features = [];
  }) words

let calculate_similarity s1 s2 =
  let tokens1 = tokenize s1 in
  let tokens2 = tokenize s2 in
  let common_tokens = List.filter ~f:(fun t1 ->
    List.exists ~f:(fun t2 -> String.equal t1.text t2.text) tokens2
  ) tokens1 in
  let similarity = float_of_int (List.length common_tokens) /.
                  float_of_int (max (List.length tokens1) (List.length tokens2)) in
  similarity

(* Advanced linguistic features *)
let extract_linguistic_features text =
  let tokens = tokenize text in
  let features = List.map ~f:(fun token ->
    let ngram_features = String.to_list token.text
      |> List.group ~break:(fun _ _ -> false)
      |> List.map ~f:(fun chars -> String.of_char_list chars) in
    (token.text, ngram_features)
  ) tokens in
  features

(* JSON serialization *)
let to_json sentence =
  sentence_to_yojson sentence

let from_json json =
  sentence_of_yojson json

(* Export functions for Python binding *)
let process_text text =
  let tokens = tokenize text in
  let features = extract_linguistic_features text in
  (tokens, features)

let compare_texts text1 text2 =
  calculate_similarity text1 text2 