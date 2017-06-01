package sg.studium.neurals.prepare

import org.apache.commons.io.IOUtils
import org.datavec.api.transform.schema.Schema
import org.junit.Test
import resource
import java.io.ByteArrayOutputStream
import kotlin.test.assertEquals


class PrepareStepsTest {

    @Test
    fun getSchemaRegression() {
        val schema = csvToSchema(resource("cpu.with.vendor.csv"))
//        println(schema.toJson())
//        println(schema.toYaml())
        assertEquals("""--- !<Schema>
columns:
- !<Categorical>
  name: "vendor"
  stateNames:
  - "adviser"
  - "amdahl"
  - "apollo"
  - "basf"
  - "bti"
  - "burroughs"
  - "c.r.d"
  - "cambex"
  - "cdc"
  - "dec"
  - "dg"
  - "formation"
  - "four-phase"
  - "gould"
  - "harris"
  - "honeywell"
  - "hp"
  - "ibm"
  - "ipl"
  - "magnuson"
  - "microdata"
  - "nas"
  - "ncr"
  - "nixdorf"
  - "perkin-elmer"
  - "prime"
  - "siemens"
  - "sperry"
  - "sratus"
  - "wang"
- !<Integer>
  name: "MYCT"
- !<Integer>
  name: "MMIN"
- !<Integer>
  name: "MMAX"
- !<Integer>
  name: "CACH"
- !<Integer>
  name: "CHMIN"
- !<Integer>
  name: "CHMAX"
- !<Integer>
  name: "class"
""", schema.toYaml())
    }

    @Test
    fun getSchemaClassification() {
        val schema = csvToSchema(resource("iris.csv"))
//        println(schema.toJson())
//        println(schema.toYaml())
        assertEquals("""--- !<Schema>
columns:
- !<Double>
  name: "sepallength"
  allowNaN: false
  allowInfinite: false
- !<Double>
  name: "sepalwidth"
  allowNaN: false
  allowInfinite: false
- !<Double>
  name: "petallength"
  allowNaN: false
  allowInfinite: false
- !<Double>
  name: "petalwidth"
  allowNaN: false
  allowInfinite: false
- !<Categorical>
  name: "class"
  stateNames:
  - "Iris-setosa"
  - "Iris-versicolor"
  - "Iris-virginica"
""", schema.toYaml())
    }

    private val cpuOneHotTxt = """vendor[adviser],vendor[amdahl],vendor[apollo],vendor[basf],vendor[bti],vendor[burroughs],vendor[c.r.d],vendor[cambex],vendor[cdc],vendor[dec],vendor[dg],vendor[formation],vendor[four-phase],vendor[gould],vendor[harris],vendor[honeywell],vendor[hp],vendor[ibm],vendor[ipl],vendor[magnuson],vendor[microdata],vendor[nas],vendor[ncr],vendor[nixdorf],vendor[perkin-elmer],vendor[prime],vendor[siemens],vendor[sperry],vendor[sratus],vendor[wang],MYCT,MMIN,MMAX,CACH,CHMIN,CHMAX,class
1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,125,256,6000,256,16,128,199
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,8000,32000,32,8,32,253
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,8000,32000,32,8,32,253
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,8000,32000,32,8,32,253
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,8000,16000,32,8,16,132
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,8000,32000,64,8,32,290
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,16000,32000,64,16,32,381
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,16000,32000,64,16,32,381
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,16000,64000,64,16,32,749
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,32000,64000,128,32,64,1238
0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,400,1000,3000,0,1,2,23
0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,400,512,3500,4,1,6,24
0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,2000,8000,65,1,8,70
0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,4000,16000,65,1,8,117
0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,350,64,64,0,1,4,15
0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,200,512,16000,0,4,32,64
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,167,524,2000,8,4,15,23
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,143,512,5000,0,7,32,29
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,143,1000,2000,0,5,16,22
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,110,5000,5000,142,8,64,124
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,143,1500,6300,0,5,32,35
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,143,3100,6200,0,5,20,39
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,143,2300,6200,0,6,64,40
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,110,3100,6200,0,6,64,45
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,320,128,6000,0,1,12,28
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,320,512,2000,4,1,3,21
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,320,256,6000,0,1,6,28
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,320,256,3000,4,1,3,22
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,320,512,5000,4,1,5,28
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,320,256,5000,4,1,6,27
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,1310,2620,131,12,24,102
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,1310,2620,131,12,24,102
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,2620,10480,30,12,24,74
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,2620,10480,30,12,24,74
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56,5240,20970,30,12,24,138
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,5240,20970,30,12,24,136
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,500,2000,8,1,4,23
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,1000,4000,8,1,5,29
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,2000,8000,8,1,5,44
0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,1000,4000,8,3,5,30
0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,1000,8000,8,3,5,41
0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,2000,16000,8,3,5,74
0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,2000,16000,8,3,6,74
0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,2000,16000,8,3,6,74
0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,1000,12000,9,3,12,54
0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,1000,8000,9,3,12,41
0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,810,512,512,8,1,1,18
0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,810,1000,5000,0,1,1,28
0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,320,512,8000,4,1,5,36
0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,200,512,8000,8,1,8,38
0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,700,384,8000,0,1,1,34
0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,700,256,2000,0,1,1,19
0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,1000,16000,16,1,3,72
0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,200,1000,8000,0,1,2,36
0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,110,1000,4000,16,1,2,30
0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,110,1000,12000,16,1,2,56
0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,220,1000,8000,16,1,2,42
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,800,256,8000,0,1,4,34
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,800,256,8000,0,1,4,34
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,800,256,8000,0,1,4,34
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,800,256,8000,0,1,4,34
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,800,256,8000,0,1,4,34
0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,125,512,1000,0,8,20,19
0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,2000,8000,64,1,38,75
0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,2000,16000,64,1,38,113
0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,2000,16000,128,1,38,157
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,90,256,1000,0,3,10,18
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,105,256,2000,0,3,10,20
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,105,1000,4000,0,3,24,28
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,105,2000,4000,8,3,19,33
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,75,2000,8000,8,3,24,47
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,75,3000,8000,8,3,48,54
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,175,256,2000,0,3,24,20
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,300,768,3000,0,6,24,23
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,300,768,3000,6,6,24,25
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,300,768,12000,6,6,24,52
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,300,768,4500,0,1,24,27
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,300,384,12000,6,1,24,50
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,300,192,768,6,6,24,18
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,180,768,12000,6,1,31,53
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,330,1000,3000,0,2,4,23
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,300,1000,4000,8,3,64,30
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,300,1000,16000,8,2,112,73
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,330,1000,2000,0,1,2,20
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,330,1000,4000,0,3,6,25
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,2000,4000,0,3,6,28
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,2000,4000,0,4,8,29
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,2000,4000,8,1,20,32
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,2000,32000,32,1,20,175
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,2000,8000,32,1,54,57
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,2000,32000,32,1,54,181
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,2000,32000,32,1,54,181
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,2000,4000,8,1,20,32
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,57,4000,16000,1,6,12,82
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,57,4000,24000,64,12,16,171
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,26,16000,32000,64,16,24,361
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,26,16000,32000,64,8,24,350
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,26,8000,32000,0,8,24,220
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,26,8000,16000,0,8,16,113
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,480,96,512,0,1,1,15
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,203,1000,2000,0,1,5,21
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,115,512,6000,16,1,6,35
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1100,512,1500,0,1,1,18
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1100,768,2000,0,1,1,20
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,600,768,2000,0,1,1,20
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,400,2000,4000,0,1,1,28
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,400,4000,8000,0,1,1,45
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,900,1000,1000,0,1,2,18
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,900,512,1000,0,1,2,17
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,900,1000,4000,4,1,2,26
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,900,1000,4000,8,1,2,28
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,900,2000,4000,0,3,6,28
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,225,2000,4000,8,3,6,31
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,225,2000,4000,8,3,6,31
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,180,2000,8000,8,1,6,42
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,185,2000,16000,16,1,6,76
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,180,2000,16000,16,1,6,76
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,225,1000,4000,2,3,6,26
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,25,2000,12000,8,1,4,59
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,25,2000,12000,16,3,5,65
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,17,4000,16000,8,6,12,101
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,17,4000,16000,32,6,12,116
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1500,768,1000,0,0,0,18
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1500,768,2000,0,0,0,20
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,800,768,2000,0,0,0,20
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,50,2000,4000,0,3,6,30
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,50,2000,8000,8,3,6,44
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,50,2000,8000,8,1,6,44
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,50,2000,16000,24,1,6,82
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,50,2000,16000,24,1,6,82
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,50,8000,16000,48,1,10,128
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,100,1000,8000,0,2,6,37
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,100,1000,8000,24,2,6,46
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,100,1000,8000,24,3,6,46
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,50,2000,16000,12,3,16,80
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,50,2000,16000,24,6,16,88
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,50,2000,16000,24,6,16,88
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,150,512,4000,0,8,128,33
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,115,2000,8000,16,1,3,46
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,115,2000,4000,2,1,5,29
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,92,2000,8000,32,1,6,53
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,92,2000,8000,32,1,6,53
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,92,2000,8000,4,1,6,41
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,75,4000,16000,16,1,6,86
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,60,4000,16000,32,1,6,95
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,60,2000,16000,64,5,8,107
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,60,4000,16000,64,5,8,117
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,50,4000,16000,64,5,10,119
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,72,4000,16000,64,8,16,120
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,72,2000,8000,16,6,8,48
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,40,8000,16000,32,8,16,126
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,40,8000,32000,64,8,24,266
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,35,8000,32000,64,8,24,270
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,38,16000,32000,128,16,32,426
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,48,4000,24000,32,8,24,151
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,38,8000,32000,64,8,24,267
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,30,16000,32000,256,16,24,603
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,112,1000,1000,0,1,4,19
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,84,1000,2000,0,1,6,21
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,56,1000,4000,0,1,6,26
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,56,2000,6000,0,1,8,35
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,56,2000,8000,0,1,8,41
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,56,4000,8000,0,1,8,47
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,56,4000,12000,0,1,8,62
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,56,4000,16000,0,1,8,78
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,38,4000,8000,32,16,32,80
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,38,4000,8000,32,16,32,80
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,38,8000,16000,64,4,8,142
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,38,8000,24000,160,4,8,281
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,38,4000,16000,128,16,32,190
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,200,1000,2000,0,1,2,21
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,200,1000,4000,0,1,4,25
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,200,2000,8000,64,1,5,67
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,250,512,4000,0,1,7,24
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,250,512,4000,0,4,7,24
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,250,1000,16000,1,1,8,64
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,160,512,4000,2,1,5,25
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,160,512,2000,2,3,8,20
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,160,1000,4000,8,1,14,29
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,160,1000,8000,16,1,14,43
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,160,2000,8000,32,1,13,53
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,240,512,1000,8,1,3,19
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,240,512,2000,8,1,5,22
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,105,2000,4000,8,3,8,31
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,105,2000,6000,16,6,16,41
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,105,2000,8000,16,4,14,47
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,52,4000,16000,32,4,12,99
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,70,4000,12000,8,6,8,67
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,59,4000,12000,32,6,12,81
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,59,8000,16000,64,12,24,149
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,26,8000,24000,32,8,16,183
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,26,8000,32000,64,12,16,275
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,26,8000,32000,128,24,32,382
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,116,2000,8000,32,5,28,56
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,50,2000,32000,24,6,26,182
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,50,2000,32000,48,26,52,227
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,50,2000,32000,112,52,104,341
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,50,4000,32000,112,52,104,360
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,30,8000,64000,96,12,176,919
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,30,8000,64000,128,12,176,978
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,180,262,4000,0,1,3,24
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,180,512,4000,0,1,3,24
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,180,262,4000,0,1,3,24
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,180,512,4000,0,1,3,24
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,124,1000,8000,0,1,8,37
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,98,1000,8000,32,2,8,50
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,125,2000,8000,0,2,14,41
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,480,512,8000,32,0,0,47
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,480,1000,4000,0,0,0,25
"""

    @Test
    fun oneHotRegressionWithSchema() {
        val schemaIn = Schema.fromYaml(IOUtils.toString(resource("cpu.with.vendor.schema.yaml")))
        val baos = ByteArrayOutputStream()
        csvToOneHot(
                resource("cpu.with.vendor.csv"),
                baos,
                schemaIn
        )
        baos.close()
//        println(schemaOut.toYaml())
//        println(baos.toString())
        assertEquals(cpuOneHotTxt, baos.toString())
    }

    @Test
    fun oneHotRegression() {
        val baos = ByteArrayOutputStream()
        csvToOneHot({ resource("cpu.with.vendor.csv") }, baos)
        baos.close()
        assertEquals(cpuOneHotTxt, baos.toString())
    }

    private val irisOneHotTxt = """sepallength,sepalwidth,petallength,petalwidth,class[Iris-setosa],class[Iris-versicolor],class[Iris-virginica]
5.1,3.5,1.4,0.2,1,0,0
4.9,3,1.4,0.2,1,0,0
4.7,3.2,1.3,0.2,1,0,0
4.6,3.1,1.5,0.2,1,0,0
5,3.6,1.4,0.2,1,0,0
5.4,3.9,1.7,0.4,1,0,0
4.6,3.4,1.4,0.3,1,0,0
5,3.4,1.5,0.2,1,0,0
4.4,2.9,1.4,0.2,1,0,0
4.9,3.1,1.5,0.1,1,0,0
5.4,3.7,1.5,0.2,1,0,0
4.8,3.4,1.6,0.2,1,0,0
4.8,3,1.4,0.1,1,0,0
4.3,3,1.1,0.1,1,0,0
5.8,4,1.2,0.2,1,0,0
5.7,4.4,1.5,0.4,1,0,0
5.4,3.9,1.3,0.4,1,0,0
5.1,3.5,1.4,0.3,1,0,0
5.7,3.8,1.7,0.3,1,0,0
5.1,3.8,1.5,0.3,1,0,0
5.4,3.4,1.7,0.2,1,0,0
5.1,3.7,1.5,0.4,1,0,0
4.6,3.6,1,0.2,1,0,0
5.1,3.3,1.7,0.5,1,0,0
4.8,3.4,1.9,0.2,1,0,0
5,3,1.6,0.2,1,0,0
5,3.4,1.6,0.4,1,0,0
5.2,3.5,1.5,0.2,1,0,0
5.2,3.4,1.4,0.2,1,0,0
4.7,3.2,1.6,0.2,1,0,0
4.8,3.1,1.6,0.2,1,0,0
5.4,3.4,1.5,0.4,1,0,0
5.2,4.1,1.5,0.1,1,0,0
5.5,4.2,1.4,0.2,1,0,0
4.9,3.1,1.5,0.1,1,0,0
5,3.2,1.2,0.2,1,0,0
5.5,3.5,1.3,0.2,1,0,0
4.9,3.1,1.5,0.1,1,0,0
4.4,3,1.3,0.2,1,0,0
5.1,3.4,1.5,0.2,1,0,0
5,3.5,1.3,0.3,1,0,0
4.5,2.3,1.3,0.3,1,0,0
4.4,3.2,1.3,0.2,1,0,0
5,3.5,1.6,0.6,1,0,0
5.1,3.8,1.9,0.4,1,0,0
4.8,3,1.4,0.3,1,0,0
5.1,3.8,1.6,0.2,1,0,0
4.6,3.2,1.4,0.2,1,0,0
5.3,3.7,1.5,0.2,1,0,0
5,3.3,1.4,0.2,1,0,0
7,3.2,4.7,1.4,0,1,0
6.4,3.2,4.5,1.5,0,1,0
6.9,3.1,4.9,1.5,0,1,0
5.5,2.3,4,1.3,0,1,0
6.5,2.8,4.6,1.5,0,1,0
5.7,2.8,4.5,1.3,0,1,0
6.3,3.3,4.7,1.6,0,1,0
4.9,2.4,3.3,1,0,1,0
6.6,2.9,4.6,1.3,0,1,0
5.2,2.7,3.9,1.4,0,1,0
5,2,3.5,1,0,1,0
5.9,3,4.2,1.5,0,1,0
6,2.2,4,1,0,1,0
6.1,2.9,4.7,1.4,0,1,0
5.6,2.9,3.6,1.3,0,1,0
6.7,3.1,4.4,1.4,0,1,0
5.6,3,4.5,1.5,0,1,0
5.8,2.7,4.1,1,0,1,0
6.2,2.2,4.5,1.5,0,1,0
5.6,2.5,3.9,1.1,0,1,0
5.9,3.2,4.8,1.8,0,1,0
6.1,2.8,4,1.3,0,1,0
6.3,2.5,4.9,1.5,0,1,0
6.1,2.8,4.7,1.2,0,1,0
6.4,2.9,4.3,1.3,0,1,0
6.6,3,4.4,1.4,0,1,0
6.8,2.8,4.8,1.4,0,1,0
6.7,3,5,1.7,0,1,0
6,2.9,4.5,1.5,0,1,0
5.7,2.6,3.5,1,0,1,0
5.5,2.4,3.8,1.1,0,1,0
5.5,2.4,3.7,1,0,1,0
5.8,2.7,3.9,1.2,0,1,0
6,2.7,5.1,1.6,0,1,0
5.4,3,4.5,1.5,0,1,0
6,3.4,4.5,1.6,0,1,0
6.7,3.1,4.7,1.5,0,1,0
6.3,2.3,4.4,1.3,0,1,0
5.6,3,4.1,1.3,0,1,0
5.5,2.5,4,1.3,0,1,0
5.5,2.6,4.4,1.2,0,1,0
6.1,3,4.6,1.4,0,1,0
5.8,2.6,4,1.2,0,1,0
5,2.3,3.3,1,0,1,0
5.6,2.7,4.2,1.3,0,1,0
5.7,3,4.2,1.2,0,1,0
5.7,2.9,4.2,1.3,0,1,0
6.2,2.9,4.3,1.3,0,1,0
5.1,2.5,3,1.1,0,1,0
5.7,2.8,4.1,1.3,0,1,0
6.3,3.3,6,2.5,0,0,1
5.8,2.7,5.1,1.9,0,0,1
7.1,3,5.9,2.1,0,0,1
6.3,2.9,5.6,1.8,0,0,1
6.5,3,5.8,2.2,0,0,1
7.6,3,6.6,2.1,0,0,1
4.9,2.5,4.5,1.7,0,0,1
7.3,2.9,6.3,1.8,0,0,1
6.7,2.5,5.8,1.8,0,0,1
7.2,3.6,6.1,2.5,0,0,1
6.5,3.2,5.1,2,0,0,1
6.4,2.7,5.3,1.9,0,0,1
6.8,3,5.5,2.1,0,0,1
5.7,2.5,5,2,0,0,1
5.8,2.8,5.1,2.4,0,0,1
6.4,3.2,5.3,2.3,0,0,1
6.5,3,5.5,1.8,0,0,1
7.7,3.8,6.7,2.2,0,0,1
7.7,2.6,6.9,2.3,0,0,1
6,2.2,5,1.5,0,0,1
6.9,3.2,5.7,2.3,0,0,1
5.6,2.8,4.9,2,0,0,1
7.7,2.8,6.7,2,0,0,1
6.3,2.7,4.9,1.8,0,0,1
6.7,3.3,5.7,2.1,0,0,1
7.2,3.2,6,1.8,0,0,1
6.2,2.8,4.8,1.8,0,0,1
6.1,3,4.9,1.8,0,0,1
6.4,2.8,5.6,2.1,0,0,1
7.2,3,5.8,1.6,0,0,1
7.4,2.8,6.1,1.9,0,0,1
7.9,3.8,6.4,2,0,0,1
6.4,2.8,5.6,2.2,0,0,1
6.3,2.8,5.1,1.5,0,0,1
6.1,2.6,5.6,1.4,0,0,1
7.7,3,6.1,2.3,0,0,1
6.3,3.4,5.6,2.4,0,0,1
6.4,3.1,5.5,1.8,0,0,1
6,3,4.8,1.8,0,0,1
6.9,3.1,5.4,2.1,0,0,1
6.7,3.1,5.6,2.4,0,0,1
6.9,3.1,5.1,2.3,0,0,1
5.8,2.7,5.1,1.9,0,0,1
6.8,3.2,5.9,2.3,0,0,1
6.7,3.3,5.7,2.5,0,0,1
6.7,3,5.2,2.3,0,0,1
6.3,2.5,5,1.9,0,0,1
6.5,3,5.2,2,0,0,1
6.2,3.4,5.4,2.3,0,0,1
5.9,3,5.1,1.8,0,0,1
"""

    @Test
    fun oneHotClassificationWithSchema() {
        val schemaIn = csvToSchema(resource("iris.csv"))
        val baos = ByteArrayOutputStream()
        csvToOneHot(
                resource("iris.csv"),
                baos,
                schemaIn
        )
        assertEquals(irisOneHotTxt, baos.toString())
    }

    @Test
    fun oneHotClassification() {
        val baos = ByteArrayOutputStream()
        csvToOneHot({ resource("iris.csv") }, baos)
        baos.close()
        assertEquals(irisOneHotTxt, baos.toString())
    }

}