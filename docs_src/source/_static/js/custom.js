/*
change the default sphinx.ext.linkcode's [source] to [Github]
*/
document.querySelectorAll(".reference.external .viewcode-link .pre").forEach(item => {
    item.innerHTML = "[Github]"
    item.style.marginRight = "3px"
})
