import path from "path"
import fs from "fs"
import util from "util"
import child_process from "child_process"

const exec = util.promisify(child_process.exec)

const downloadSorter = async () => {
    const sorterPath = path.join(__dirname, "../sorting")
    if (!fs.existsSync(sorterPath)) fs.mkdirSync(sorterPath, {recursive: true})
    const configPath = path.join(sorterPath, "config.json")
    const modelPath = path.join(sorterPath, "model.safetensors")
    const preprocessorPath = path.join(sorterPath, "preprocessor_config.json")
    if (!fs.existsSync(configPath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/sorter/config.json`).then((r) => r.json())
        fs.writeFileSync(configPath, JSON.stringify(data, null, 4))
    }
    if (!fs.existsSync(preprocessorPath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/sorter/preprocessor_config.json`).then((r) => r.json())
        fs.writeFileSync(preprocessorPath, JSON.stringify(data, null, 4))
    }
    if (!fs.existsSync(modelPath)) {
        console.log("Downloading sorting model...")
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/sorter/model.safetensors`).then((r) => r.arrayBuffer())
        fs.writeFileSync(modelPath, Buffer.from(data))
        console.log("Done!")
    }
}

const downloadQualifier = async () => {
    const qualifierPath = path.join(__dirname, "../adapter")
    if (!fs.existsSync(qualifierPath)) fs.mkdirSync(qualifierPath, {recursive: true})
    const configPath = path.join(qualifierPath, "adapter_config.json")
    const modelPath = path.join(qualifierPath, "adapter_model.safetensors")
    const templatePath = path.join(qualifierPath, "chat_template.jinja")
    const preprocessorPath = path.join(qualifierPath, "preprocessor_config.json")
    const tokenizerConfigPath = path.join(qualifierPath, "tokenizer_config.json")
    const tokenizerPath = path.join(qualifierPath, "tokenizer.json")
    if (!fs.existsSync(configPath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/qualifiers/adapter_config.json`).then((r) => r.json())
        fs.writeFileSync(configPath, JSON.stringify(data, null, 4))
    }
    if (!fs.existsSync(templatePath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/qualifiers/chat_template.jinja`).then((r) => r.text())
        fs.writeFileSync(templatePath, data)
    }
    if (!fs.existsSync(preprocessorPath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/qualifiers/preprocessor_config.json`).then((r) => r.json())
        fs.writeFileSync(preprocessorPath, JSON.stringify(data, null, 4))
    }
    if (!fs.existsSync(tokenizerConfigPath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/qualifiers/tokenizer_config.json`).then((r) => r.json())
        fs.writeFileSync(tokenizerConfigPath, JSON.stringify(data, null, 4))
    }
    if (!fs.existsSync(tokenizerPath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/qualifiers/tokenizer.json`).then((r) => r.json())
        fs.writeFileSync(tokenizerPath, JSON.stringify(data, null, 4))
    }
    if (!fs.existsSync(modelPath)) {
        console.log("Downloading qualifier model...")
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/qualifiers/adapter_model.safetensors`).then((r) => r.arrayBuffer())
        fs.writeFileSync(modelPath, Buffer.from(data))
        console.log("Done!")
    }
}

const downloadWDTagger = async () => {
    const wdTaggerPath = path.join(__dirname, "../tagger")
    if (!fs.existsSync(wdTaggerPath)) fs.mkdirSync(wdTaggerPath, {recursive: true})
    const configPath = path.join(wdTaggerPath, "config.json")
    const modelPath = path.join(wdTaggerPath, "model.safetensors")
    const csvPath = path.join(wdTaggerPath, "selected_tags.csv")
    if (!fs.existsSync(configPath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/wdtagger/config.json`).then((r) => r.json())
        fs.writeFileSync(configPath, JSON.stringify(data, null, 4))
    }
    if (!fs.existsSync(csvPath)) {
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/wdtagger/selected_tags.csv`).then((r) => r.text())
        fs.writeFileSync(csvPath, data)
    }
    if (!fs.existsSync(modelPath)) {
        console.log("Downloading waifu diffusion tagger...")
        const data = await fetch(`https://huggingface.co/Moebits/anime-models/resolve/main/wdtagger/model.safetensors`).then((r) => r.arrayBuffer())
        fs.writeFileSync(modelPath, Buffer.from(data))
        console.log("Done!")
    }
}

const downloadModels = async () => {
    await downloadSorter()
    await downloadQualifier()
    await downloadWDTagger()
}

const sortImages = async (folder: string) => {
    const scriptPath = path.join(__dirname, "../sort_images.py")
    
    await new Promise<void>((resolve, reject) => {
        const process = child_process.spawn("python3", ["-u", scriptPath, folder], 
            {stdio: ["ignore", "pipe", "pipe"]})

        process.stdout.on("data", (data: Buffer) => {
            process.stdout && console.log(data.toString().trimEnd())
        })

        process.stderr.on("data", (data: Buffer) => {
            console.error(data.toString().trimEnd())
        })

        process.on("error", reject)
        process.on("close", resolve)
    })

    return path.join(folder, "good")
}

const predictRelationships = async (folder: string) => {
    const scriptPath = path.join(__dirname, "../predict_relations.py")
    
    await new Promise<void>((resolve, reject) => {
        const process = child_process.spawn("python3", ["-u", scriptPath, folder], 
            {stdio: ["ignore", "pipe", "pipe"]})

        process.stdout.on("data", (data: Buffer) => {
            process.stdout && console.log(data.toString().trimEnd())
        })

        process.stderr.on("data", (data: Buffer) => {
            console.error(data.toString().trimEnd())
        })

        process.on("error", reject)
        process.on("close", resolve)
    })


    const jsonPath = path.join(folder, "predictions.json")
    const json = JSON.parse(fs.readFileSync(jsonPath).toString())
    fs.unlinkSync(jsonPath)
    return json
}

const extractSpecialQualifiers = async (folder: string, image: string) => {
    const imagePath = path.join(folder, image)
    const scriptPath = path.join(__dirname, "../wdtagger.py")
    const wdTaggerPath = path.join(__dirname, "../tagger")
    const command = `python3 "${scriptPath}" -i "${imagePath}" -m "${wdTaggerPath}"`
    const str = await exec(command).then((s: any) => s.stdout).catch((e: any) => e.stderr)
    const json = JSON.parse(str.match(/{.*?}/gm)?.[0]) as {tags: string[], characters: string[]}

    const predictedQualifiers = [] as string[]
    const tagArr = json.tags

    if (tagArr.includes("comic")) {
        if (tagArr.includes("monochrome") || tagArr.includes("greyscale")) {
            predictedQualifiers.push("comic")
        } else {
            predictedQualifiers.push("color-comic")
        }
    }
    if (tagArr.includes("english_text") || tagArr.includes("japanese_text")
        || tagArr.includes("chinese_text") || tagArr.includes("korean_text")
        || tagArr.includes("speech_bubble") || tagArr.includes("text_focus")) {
        predictedQualifiers.push("text")
    }
    if (tagArr.includes("chibi")) predictedQualifiers.push("chibi")
    if (tagArr.includes("pixel_art")) predictedQualifiers.push("pixel")
    if (tagArr.includes("3d")) predictedQualifiers.push("3d")
    if (tagArr.includes("dakimakura")) predictedQualifiers.push("daki")
    if (tagArr.includes("sketch")) predictedQualifiers.push("sketch")
    if (tagArr.includes("lineart")) predictedQualifiers.push("lineart")
    if (tagArr.includes("ad")) predictedQualifiers.push("promo")
    if (tagArr.includes("fanbox_username")) predictedQualifiers.push("fanbox")
    if (tagArr.includes("patreon_username")) predictedQualifiers.push("patreon")

    return predictedQualifiers
}

const removeDuplicates = <T>(array: T[]) => {
    const set = new Set<string>()
    return array.filter(item => {
        const serialized = JSON.stringify(item)
        if (set.has(serialized)) {
            return false
        } else {
            set.add(serialized)
            return true
        }
    })
}

const predictQualifiers = async (folder: string, relationships: {[key: string]: string}) => {
    const files = fs.readdirSync(folder).filter(file => file !== ".DS_Store")
    const obj: {[id: string]: string[]} = {}

    for (const file of files) {
        const name = path.parse(file).name
        const firstUnderscore = name.indexOf("_")
        const id = firstUnderscore === -1 ? name : name.substring(0, firstUnderscore)
        if (!obj[id]) obj[id] = []
        obj[id].push(file)
    }

    let i = 0
    for (const [id, groupFiles] of Object.entries(obj)) {
        let predictedQualifiers = await extractSpecialQualifiers(folder, groupFiles[0])

        if (groupFiles.length > 1) {
            let lastQualifiers = await extractSpecialQualifiers(folder, groupFiles[groupFiles.length - 1])
            predictedQualifiers = removeDuplicates([...predictedQualifiers, ...lastQualifiers])
        }

        for (const file of groupFiles) {
            let predictedRelation = relationships[file] || "_p"
            if (groupFiles.length === 1) predictedRelation = "_p"
            predictedRelation = predictedRelation.replace("_", "")

            const {name, ext} = path.parse(file)
            const match = name.match(/(_s|_p|_g!?|_c!?)(\d+)?$/)

            const qualifierSuffix = predictedQualifiers.length ? `_${predictedQualifiers.join("_")}` : ""
            let newName = `${name}${qualifierSuffix}_${predictedRelation}0`

            if (match) {
                const num = match[2] ?? "0"
                const baseName = name.replace(/(_s|_p|_g!?|_c!?)(\d+)?$/, "")
                newName = `${baseName}${qualifierSuffix}_${predictedRelation}${num}`
            }

            const src = path.join(folder, file)
            const dest = path.join(folder, `${newName}${ext}`)
            fs.renameSync(src, dest)
            console.log(`${++i} / ${files.length - 1}: ${file} -> ${newName}${ext}`)
        }
    }
}

const start = async () => {
    await downloadModels()
    let folder = ""

    folder = await sortImages(folder)
    const json = await predictRelationships(folder)
    fs.writeFileSync("current.json", JSON.stringify(json, undefined, 4))
    await predictQualifiers(folder, json)
}

start()