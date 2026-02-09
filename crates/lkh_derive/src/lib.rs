mod lkh_display;

use proc_macro::TokenStream;

#[proc_macro_derive(LkhDisplay, attributes(lkh))]
pub fn derive_lkh_display(input: TokenStream) -> TokenStream {
    lkh_display::derive_lkh_display_inner(input)
}
