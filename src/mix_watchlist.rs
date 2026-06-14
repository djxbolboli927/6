use std::collections::HashSet;

pub fn load_mix_watchlist(path: &str) -> HashSet<String> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[mix_watchlist] cannot read {path}: {e} — using empty watchlist");
            return HashSet::new();
        }
    };
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[mix_watchlist] parse error {path}: {e}");
            return HashSet::new();
        }
    };
    let mut pubkeys = HashSet::new();
    walk_json(&json, &mut pubkeys);
    eprintln!("[mix_load] path={path} accounts={}", pubkeys.len());
    pubkeys
}

fn walk_json(val: &serde_json::Value, out: &mut HashSet<String>) {
    match val {
        serde_json::Value::String(s) => {
            if looks_like_pubkey(s) { out.insert(s.clone()); }
        }
        serde_json::Value::Array(arr) => {
            for v in arr { walk_json(v, out); }
        }
        serde_json::Value::Object(obj) => {
            for (k, v) in obj {
                if looks_like_pubkey(k) { out.insert(k.clone()); }
                walk_json(v, out);
            }
        }
        _ => {}
    }
}

fn looks_like_pubkey(s: &str) -> bool {
    let len = s.len();
    if len < 32 || len > 44 { return false; }
    bs58::decode(s).into_vec().map(|v| v.len() == 32).unwrap_or(false)
}
