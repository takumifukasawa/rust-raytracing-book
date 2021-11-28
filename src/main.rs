
mod rayt;

mod code_001;
mod code_002;

fn run(id: &str) {
    match id {
        "001" => code_001::run(),
        "002" => code_002::run(),
        _ => println!("not found code_{}", id)
    }
}

fn main() {
    let mut no = "0";
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        no = &args[1];
    }
    println!("run code_{}", no);
    run(no);
}
