"""poke-engine Python 바인딩에 커스텀 함수 추가.

get_all_options(), is_terminal(), index-based switch parsing 패치.
"""

import re

LIB_RS = "poke-engine/poke-engine-py/src/lib.rs"

with open(LIB_RS, "r") as f:
    content = f.read()

# 이미 패치 적용됨
if "get_all_options" in content:
    print("patch_poke_engine: already patched, skipping")
    exit(0)

# 1. parse_move_choice 함수 + generate_instructions 교체
old_gen_instr = '''#[pyfunction]
fn generate_instructions(
    py_state: PyState,
    side_one_move: String,
    side_two_move: String,
) -> PyResult<Vec<PyStateInstructions>> {
    let (s1_move, s2_move);
    let mut state: State = py_state.into();
    match MoveChoice::from_string(&side_one_move, &state.side_one) {
        Some(m) => s1_move = m,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s1: {}",
                side_one_move
            )))
        }
    }
    match MoveChoice::from_string(&side_two_move, &state.side_two) {
        Some(m) => s2_move = m,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s2: {}",
                side_two_move
            )))
        }
    }
    let instructions = generate_instructions_from_move_pair(&mut state, &s1_move, &s2_move, true);
    let py_instructions = instructions.iter().map(|i| i.clone().into()).collect();

    Ok(py_instructions)
}'''

new_gen_instr = '''/// Parse a move string into MoveChoice, supporting "switch <index>" format
fn parse_move_choice(s: &str, side: &Side) -> Option<MoveChoice> {
    let lower = s.to_lowercase();
    // "switch 0"~"switch 5" → index-based switch
    if lower.starts_with("switch ") {
        let idx_str = lower.strip_prefix("switch ").unwrap().trim();
        if let Ok(_) = idx_str.parse::<u8>() {
            let pkmn_index = PokemonIndex::deserialize(idx_str);
            return Some(MoveChoice::Switch(pkmn_index));
        }
    }
    MoveChoice::from_string(&lower, side)
}

#[pyfunction]
fn generate_instructions(
    py_state: PyState,
    side_one_move: String,
    side_two_move: String,
) -> PyResult<Vec<PyStateInstructions>> {
    let (s1_move, s2_move);
    let mut state: State = py_state.into();
    match parse_move_choice(&side_one_move, &state.side_one) {
        Some(m) => s1_move = m,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s1: {}",
                side_one_move
            )))
        }
    }
    match parse_move_choice(&side_two_move, &state.side_two) {
        Some(m) => s2_move = m,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s2: {}",
                side_two_move
            )))
        }
    }
    let instructions = generate_instructions_from_move_pair(&mut state, &s1_move, &s2_move, true);
    let py_instructions = instructions.iter().map(|i| i.clone().into()).collect();

    Ok(py_instructions)
}'''

content = content.replace(old_gen_instr, new_gen_instr)

# 2. pymodule에 새 함수 등록 + get_all_options/is_terminal 추가
old_module = '''#[pymodule]
#[pyo3(name = "poke_engine")]
fn py_poke_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_damage, m)?)?;
    m.add_function(wrap_pyfunction!(generate_instructions, m)?)?;'''

new_module_with_fns = '''#[pyfunction]
fn get_all_options(py_state: PyState) -> PyResult<(Vec<String>, Vec<String>)> {
    let state: State = py_state.into();
    let (s1_options, s2_options) = state.root_get_all_options();
    let s1_strings: Vec<String> = s1_options
        .iter()
        .map(|m| movechoice_to_string_indexed(&state.side_one, m))
        .collect();
    let s2_strings: Vec<String> = s2_options
        .iter()
        .map(|m| movechoice_to_string_indexed(&state.side_two, m))
        .collect();
    Ok((s1_strings, s2_strings))
}

fn movechoice_to_string_indexed(side: &Side, move_choice: &MoveChoice) -> String {
    match move_choice {
        MoveChoice::Switch(idx) => {
            format!("switch {}", idx.serialize())
        }
        MoveChoice::None => "none".to_string(),
        _ => move_choice.to_string(side),
    }
}

#[pyfunction]
fn is_terminal(py_state: PyState) -> PyResult<(bool, Option<String>)> {
    let state: State = py_state.into();
    let s1_alive = state.side_one.pokemon.into_iter().any(|p| p.hp > 0);
    let s2_alive = state.side_two.pokemon.into_iter().any(|p| p.hp > 0);
    match (s1_alive, s2_alive) {
        (false, false) => Ok((true, Some("tie".to_string()))),
        (true, false) => Ok((true, Some("side_one".to_string()))),
        (false, true) => Ok((true, Some("side_two".to_string()))),
        (true, true) => Ok((false, None)),
    }
}

#[pymodule]
#[pyo3(name = "poke_engine")]
fn py_poke_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_damage, m)?)?;
    m.add_function(wrap_pyfunction!(generate_instructions, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_options, m)?)?;
    m.add_function(wrap_pyfunction!(is_terminal, m)?)?;'''

content = content.replace(old_module, new_module_with_fns)

with open(LIB_RS, "w") as f:
    f.write(content)

print("patch_poke_engine: patched successfully")
