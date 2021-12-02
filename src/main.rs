mod rayt;

mod code_101;
mod code_102;
mod code_103;
mod code_104;
mod code_105;
mod code_106;
mod code_107;
mod code_108;
mod code_109;
mod code_109_2;
mod code_110;
mod code_111;
mod code_112;
mod code_113;
mod code_114;
mod code_115;
mod code_116;
mod code_117;
mod code_201;
mod code_202;
mod code_203;

fn run(id: &str) {
    match id {
        "101" => code_101::run(),
        "102" => code_102::run(),
        "103" => code_103::run(),
        "104" => code_104::run(),
        "105" => code_105::run(),
        "106" => code_106::run(),
        "107" => code_107::run(),
        "108" => code_108::run(),
        "109" => code_109::run(),
        "109_2" => code_109_2::run(),
        "110" => code_110::run(),
        "111" => code_111::run(),
        "112" => code_112::run(),
        "113" => code_113::run(),
        "114" => code_114::run(),
        "115" => code_115::run(),
        "116" => code_116::run(),
        "117" => code_117::run(),
        "201" => code_201::run(),
        "202" => code_202::run(),
        "203" => code_203::run(),
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
