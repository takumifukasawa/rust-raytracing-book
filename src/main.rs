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
mod code_204;
mod code_205;
mod code_206;
mod code_207;
mod code_208;
mod code_209;
mod code_301;
mod code_302;
mod code_303;
mod code_304;
mod code_305;
mod code_306;
mod code_307;
mod code_308;
mod code_309;
mod code_310;
mod code_311;

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
        "204" => code_204::run(),
        "205" => code_205::run(),
        "206" => code_206::run(),
        "207" => code_207::run(),
        "208" => code_208::run(),
        "209" => code_209::run(),
        "301" => code_301::run(),
        "302" => code_302::run(),
        "303" => code_303::run(),
        "304" => code_304::run(),
        "305" => code_305::run(),
        "306" => code_306::run(),
        "307" => code_307::run(),
        "308" => code_308::run(),
        "309" => code_309::run(),
        "310" => code_310::run(),
        "311" => code_311::run(),
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
