cols = []
ruta = ""

fileCSV = () =>{
    file = document.getElementById("csv").files[0].name
    eel.getFile(file)
    ruta = file
}

eel.expose(receive_err_file)
function receive_err_file(){
    alert("Archivo no valido.")
}

sendOpteions = async () =>{ 
    series = ""
    cols.forEach((e,i)=>{
        series += document.getElementById(e).checked? (series=="" ? "": ",") + (i+1): ""
    })
    if(series!=""){
        perc_train = parseInt(document.getElementById("perc_train").value)
        hm_day_more = parseInt(document.getElementById("hm_day_more").value)
        epochs = parseInt(document.getElementById("epochs").value)
        layers = parseInt(document.getElementById("layers").value)
        batch = parseInt(document.getElementById("batch").value)
        eel.sendOptions(ruta,series,perc_train,hm_day_more,epochs,layers,batch)
        document.getElementById("app_main").style.display="none"
        document.getElementById("app_load").style.display=""
    }
    else
        alert("Seleccione las columnas.")
}

eel.expose(receive_colums)
function receive_colums(c){
    cols = c
    html = ""
    c.forEach(e => {
        html+= `
        <div class="col-md-2">
            <div class="form-check">
                <input class="form-check-input" type="checkbox" id="`+e+`">
                <label class="form-check-label" for="`+e+`">
                `+e+`
                </label>
            </div>
        </div>`
    });
    document.getElementById('colums').innerHTML=html
    document.getElementById("filediv").style.display="none"
    document.getElementById("options").style.display=""
}

eel.expose(receive_err)
function receive_err(e){
    alert(e)
    document.getElementById("csv").value=""
    document.getElementById('colums').innerHTML=""
    document.getElementById("filediv").style.display=""
    document.getElementById("options").style.display="none"
    document.getElementById("app_main").style.display=""
    document.getElementById("app_load").style.display="none"
}

eel.expose(receive_check)
function receive_check(e){
    document.getElementById("csv").value=""
    document.getElementById('colums').innerHTML=""
    document.getElementById("filediv").style.display=""
    document.getElementById("options").style.display="none"
    document.getElementById("app_main").style.display=""
    document.getElementById("app_load").style.display="none"
}