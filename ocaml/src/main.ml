open Core
open Core_kernel
open Linguistic

let process_command text =
  let tokens, features = process_text text in
  let json = Yojson.Safe.to_string (to_json tokens) in
  print_endline json

let compare_command text1 text2 =
  let similarity = compare_texts text1 text2 in
  Printf.printf "%.4f\n" similarity

let () =
  let open Command in
  let spec = Spec.(
    empty
    +> flag "-process" (optional string) ~doc:"Process text for linguistic analysis"
    +> flag "-compare" (optional (t2 string string)) ~doc:"Compare two texts for similarity"
  ) in
  let command = basic ~summary:"Semantic Doppelgängers OCaml CLI"
    spec
    (fun process compare () ->
      match process, compare with
      | Some text, None -> process_command text
      | None, Some (text1, text2) -> compare_command text1 text2
      | _ -> failwith "Invalid command. Use -process or -compare")
  in
  Command.run command 