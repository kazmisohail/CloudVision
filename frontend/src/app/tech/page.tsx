import { Database, Layers, Cpu, Code } from 'lucide-react';
import Image from 'next/image';

export default function TechPage() {
    return (
        <main className="min-h-screen bg-[#0f172a] overflow-x-hidden relative">
            {/* Background Elements */}
            <div className="fixed inset-0 z-0">
                <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')] bg-cover bg-center opacity-50"></div>
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#0f172a]/50 to-[#0f172a]"></div>
            </div>

            <div className="relative z-10 w-full max-w-6xl mx-auto px-4 py-24 space-y-16">
                {/* Header */}
                <div className="text-center space-y-4">
                    <h1 className="text-4xl md:text-5xl font-bold text-white">Technical Architecture</h1>
                    <p className="text-cyan-400 max-w-2xl mx-auto">
                        A deep dive into the Attention U-Net model, L8CCA dataset, and our scalable full-stack pipeline.
                    </p>
                </div>

                {/* Bento Grid Features */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Feature 1: Model */}
                    <div className="md:col-span-2 bg-slate-900/60 backdrop-blur-md border border-white/10 rounded-2xl p-8 hover:bg-slate-800/70 transition-colors shadow-xl">
                        <div className="flex items-start justify-between mb-6">
                            <div className="p-3 bg-cyan-500/20 rounded-lg">
                                <Cpu className="h-8 w-8 text-cyan-400" />
                            </div>
                            <span className="text-xs font-mono text-gray-500">MODEL ARCHITECTURE</span>
                        </div>
                        <h3 className="text-2xl font-semibold text-white mb-4">Attention U-Net</h3>
                        <p className="text-gray-300 leading-relaxed">
                            Our core segmentation engine utilizes an Attention U-Net architecture. Unlike standard U-Nets, it incorporates
                            <span className="text-cyan-400"> Attention Gates</span> that automatically learn to focus on target structures
                            (clouds and shadows) of varying shapes and sizes. This suppresses irrelevant regions in the input image while
                            highlighting salient features useful for the specific task.
                        </p>
                        <div className="mt-6 w-full bg-slate-900/50 rounded-lg border border-white/5 overflow-hidden">
                            <Image
                                src="/model_architecture.png"
                                alt="Attention U-Net Architecture"
                                width={0}
                                height={0}
                                sizes="100vw"
                                style={{ width: '100%', height: 'auto' }}
                            />
                        </div>
                    </div>

                    {/* Feature 2: Dataset */}
                    <div className="bg-slate-900/60 backdrop-blur-md border border-white/10 rounded-2xl p-8 hover:bg-slate-800/70 transition-colors shadow-xl">
                        <div className="flex items-start justify-between mb-6">
                            <div className="p-3 bg-purple-500/20 rounded-lg">
                                <Database className="h-8 w-8 text-purple-400" />
                            </div>
                            <span className="text-xs font-mono text-gray-500">DATASET</span>
                        </div>
                        <h3 className="text-2xl font-semibold text-white mb-4">L8CCA</h3>
                        <p className="text-gray-300 leading-relaxed text-sm">
                            Trained on the Landsat 8 Cloud Cover Assessment (L8CCA) dataset. It provides diverse biomes (Barren, Forest, Grass, Shrubland, Snow, Urban, Water, Wetlands) ensuring robust generalization.
                        </p>
                        <ul className="mt-4 space-y-2 text-sm text-gray-400">
                            <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-purple-400"></div> 8-Channel Multispectral</li>
                            <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-purple-400"></div> Manually Annotated Masks</li>
                        </ul>
                    </div>

                    {/* Feature 3: Preprocessing */}
                    <div className="bg-slate-900/60 backdrop-blur-md border border-white/10 rounded-2xl p-8 hover:bg-slate-800/70 transition-colors shadow-xl">
                        <div className="flex items-start justify-between mb-6">
                            <div className="p-3 bg-blue-500/20 rounded-lg">
                                <Layers className="h-8 w-8 text-blue-400" />
                            </div>
                            <span className="text-xs font-mono text-gray-500">PIPELINE</span>
                        </div>
                        <h3 className="text-2xl font-semibold text-white mb-4">Preprocessing</h3>
                        <p className="text-gray-300 leading-relaxed text-sm">
                            Input RGB images are resized to 256x256 and normalized. To match the model's 8-channel input expectation,
                            we employ a zero-padding strategy for the missing spectral bands.
                        </p>
                    </div>

                    {/* Feature 4: Stack */}
                    <div className="md:col-span-2 bg-slate-900/60 backdrop-blur-md border border-white/10 rounded-2xl p-8 hover:bg-slate-800/70 transition-colors shadow-xl">
                        <div className="flex items-start justify-between mb-6">
                            <div className="p-3 bg-green-500/20 rounded-lg">
                                <Code className="h-8 w-8 text-green-400" />
                            </div>
                            <span className="text-xs font-mono text-gray-500">TECH STACK</span>
                        </div>
                        <h3 className="text-2xl font-semibold text-white mb-4">Full-Stack Implementation</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="p-4 bg-slate-900/80 rounded-lg border border-white/5 text-center">
                                <div className="font-semibold text-white">Next.js 14</div>
                                <div className="text-xs text-gray-500">Frontend</div>
                            </div>
                            <div className="p-4 bg-slate-900/80 rounded-lg border border-white/5 text-center">
                                <div className="font-semibold text-white">Django 5.0</div>
                                <div className="text-xs text-gray-500">Backend</div>
                            </div>
                            <div className="p-4 bg-slate-900/80 rounded-lg border border-white/5 text-center">
                                <div className="font-semibold text-white">TensorFlow</div>
                                <div className="text-xs text-gray-500">Inference</div>
                            </div>
                            <div className="p-4 bg-slate-900/80 rounded-lg border border-white/5 text-center">
                                <div className="font-semibold text-white">Gemini 2.5</div>
                                <div className="text-xs text-gray-500">GenAI</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
