mod rayt;

mod code_101;
mod code_102;
mod code_103;

fn run(id: &str) {
    match id {
        "101" => code_101::run(),
        "102" => code_102::run(),
        "103" => code_103::run(),
        _ => println!("not found code_{}", id),
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
