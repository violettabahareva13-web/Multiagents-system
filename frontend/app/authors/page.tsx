import Image, { type StaticImageData } from "next/image";
import Link from "next/link";

import dimaPhoto from "@/faces/dima.webp";
import igorPhoto from "@/faces/igor.webp";
import lenaPhoto from "@/faces/Lena.webp";
import serjPhoto from "@/faces/serj.webp";
import violettaPhoto from "@/faces/Violetta.webp";

type AuthorCard = {
  fileName: string;
  name: string;
  subtitle: string;
  description: string;
  photo: StaticImageData;
};

const AUTHORS: AuthorCard[] = [
  {
    fileName: "igor.webp",
    name: "Игорь",
    subtitle: "Data Scientist",
    description:
      "Автор идеи и архитектор проекта. Спроектировал ключевые принципы работы системы и её техническую структуру. Обладает выдающимися навыками программирования, реализовал критически важные компоненты и задал высокий инженерный стандарт разработки.",
    photo: igorPhoto,
  },
  {
    fileName: "Lena.webp",
    name: "Елена Якунова",
    subtitle: "Data Scientist",
    description:
      "Эксперт по работе с данными. Глубоко понимает структуру БД и архитектуру хранения. Отвечала за организацию данных, настройку обмена и обеспечение корректной и стабильной работы всей системы.",
    photo: lenaPhoto,
  },
  {
    fileName: "Violetta.webp",
    name: "Виолетта Бахарева",
    subtitle: "Data Scientist",
    description:
      "Опытный разработчик. Проводила детальное код-ревью, помогала повышать качество кода. Консультировала по инструментам и объясняла принципы их работы, усилив инженерную зрелость проекта.",
    photo: violettaPhoto,
  },
  {
    fileName: "serj.webp",
    name: "Сергей Белькин",
    subtitle: "Data Scientist",
    description:
      "Data Scientist с сильной инженерной экспертизой. Отвечал за разработку ключевых компонентов, писал чистый и эффективный код. Участвовал в создании фронтенда и серверной логики, обеспечив целостность системы.",
    photo: serjPhoto,
  },
  {
    fileName: "dima.webp",
    name: "Дмитрий Павлов",
    subtitle: "ML-инженер",
    description:
      "Специалист по AI-агентам и FastAPI. Отвечал за архитектуру интеллектуальной части системы и backend-сервисов. Реализовал взаимодействие компонентов и обеспечил стабильную работу API.",
    photo: dimaPhoto,
  },
];

export default function AuthorsPage() {
  return (
    <main className="relative min-h-screen overflow-hidden text-white">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(1000px_circle_at_10%_5%,rgba(56,189,248,0.18),transparent_55%),radial-gradient(1100px_circle_at_85%_20%,rgba(168,85,247,0.18),transparent_55%),linear-gradient(to_bottom,rgba(2,6,23,0.96),rgba(2,6,23,0.88),rgba(2,6,23,0.98))]" />
      </div>

      <div className="mx-auto w-full max-w-[1600px] px-6 py-10">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">Авторы проекта</h1>
            <p className="mt-2 text-sm text-white/65">Команда, которая спроектировала и реализовала систему.</p>
          </div>
          <Link
            href="/"
            className="inline-flex h-10 items-center rounded-lg border border-white/15 bg-white/5 px-4 text-sm text-white/90 transition hover:-translate-y-0.5 hover:bg-white/10"
          >
            На главную
          </Link>
        </div>

        <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
          {AUTHORS.map((author) => (
            <article key={author.fileName} className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
              <div className="relative mb-4 overflow-hidden rounded-xl border border-white/10 bg-black/20">
                <Image
                  src={author.photo}
                  alt={author.name}
                  className="h-[300px] w-full object-cover object-top"
                  sizes="(max-width: 640px) 100vw, (max-width: 1280px) 50vw, 33vw"
                  priority={author.fileName === "igor.webp"}
                />
              </div>

              <div className="space-y-2">
                <div>
                  <h2 className="text-xl font-semibold leading-tight">{author.name}</h2>
                  <div className="mt-1 text-sm text-cyan-200/90">{author.subtitle}</div>
                </div>
                <p className="text-sm leading-relaxed text-white/80">{author.description}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </main>
  );
}
