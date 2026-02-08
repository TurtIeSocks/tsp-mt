mod as_lkh;
mod cli_options;
mod cli_value;
mod kv_display;
mod new;
mod timer;
mod utils;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn timer(attr: TokenStream, item: TokenStream) -> TokenStream {
    timer::timer_inner(attr, item)
}

#[proc_macro_derive(CliValue, attributes(cli_value, cli))]
pub fn derive_cli_value(item: TokenStream) -> TokenStream {
    cli_value::derive_cli_value_inner(item)
}

#[proc_macro_derive(CliOptions, attributes(cli))]
pub fn derive_cli_options(item: TokenStream) -> TokenStream {
    cli_options::derive_cli_options_inner(item)
}

#[proc_macro_derive(KvDisplay, attributes(kv))]
pub fn derive_kv_display(item: TokenStream) -> TokenStream {
    kv_display::derive_kv_display_inner(item)
}

#[proc_macro_derive(New)]
pub fn derive_new(input: TokenStream) -> TokenStream {
    new::derive_new_inner(input)
}

#[proc_macro_derive(AsLkh, attributes(lkh))]
pub fn derive_as_lkh(input: TokenStream) -> TokenStream {
    as_lkh::derive_as_lkh_inner(input)
}
